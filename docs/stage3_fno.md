# Stage 3: conditioned one-step Fourier neural operator

Stage 3 learns the fixed-horizon map `(u(t), cx, cy, nu) -> u(t+0.1)` from
the Stage 2 development archive. It uses the existing Stage 1 numerical
solutions as labels without changing the solver, dataset generator, or
archive. This is a small in-distribution development baseline, not a claim
of continuum accuracy or general operator-learning performance.

## Install and run

From the repository root, use the existing virtual environment:

```bash
.venv/bin/python -m pip install -e '.[test,ml]'
.venv/bin/python -m pytest -q
.venv/bin/python -m examples.train_fno --output runs/stage3_baseline
```

The archive defaults to `data/dev/trajectories.npz`. `--archive` and
`--output` accept alternative paths. The default full training budget is
100 epochs, adjustable with `--epochs`; the CPU thread count defaults to
4, adjustable with `--threads`. Output directories must not already exist,
preventing accidental replacement of a previous run. All generated run
artifacts are under the ignored `/runs/` directory by default.

PyTorch is an optional ML dependency; Stage 1/2 remain usable with NumPy.
ML-dependent tests skip if PyTorch is not installed. For the complete suite,
install the `ml` extra. The reported baseline uses Python 3.10.8, NumPy 2.2.6,
and PyTorch 2.14.0 on CPU with four intra-op threads.

## Data and trajectory split

The loader first uses the existing Stage 2 schema validator, then requires
`fields.shape == (64,11,64,64)`, `coefficients.shape == (64,3)`, and snapshot
differences of 0.1 (absolute tolerance `1e-12`). Coefficients are ordered
`cx, cy, nu`. Incompatible datasets are rejected rather than silently
changing the learning problem.

A local `Generator(PCG64(2027))` permutes the 64 trajectory IDs. The first
48 form training, the next 8 validation, and the final 8 test. Whole
trajectories are assigned before pair extraction:

| Split | Trajectories | Adjacent pairs |
| --- | ---: | ---: |
| Training | 48 | 480 |
| Validation | 8 | 80 |
| Test | 8 | 80 |

The exact IDs, in permutation order, are:

```text
train = [0, 50, 21, 20, 38, 10, 45, 59, 29, 31, 35, 8,
         58, 46, 13, 34, 30, 33, 61, 44, 11, 32, 7, 53,
         26, 52, 42, 37, 6, 12, 36, 54, 2, 23, 18, 39,
         15, 24, 1, 63, 4, 62, 57, 19, 56, 27, 55, 14]
validation = [60, 41, 5, 9, 48, 25, 43, 3]
test = [51, 16, 49, 17, 40, 47, 22, 28]
```

The Dataset stores only `(trajectory_id,k)` indices for `k=0,...,9` and
constructs each pair on demand. Each example contains normalized
`fields[i,k]`, broadcast normalized coefficients, and normalized
`fields[i,k+1]`. It retains the trajectory/time indices for diagnostics.
The archive is loaded into memory but is never written by Stage 3.

| Quantity | Per example | Batch |
| --- | --- | --- |
| Input channels `[u,cx,cy,nu]` | `(4,64,64)` | `(B,4,64,64)` |
| Target field | `(1,64,64)` | `(B,1,64,64)` |
| Predicted field | `(1,64,64)` | `(B,1,64,64)` |

Training uses float32 tensors; the archive remains float64. DataLoaders
use batch size 8, zero workers, and no dropped final batch. Only training
is shuffled, using a dedicated seeded PyTorch generator. No pair or
snapshot crosses a split boundary. There are 64 independently sampled
trajectories, not 640 independent simulations.

## Global normalization

All statistics are computed in float64 from the training split only.
The shared field mean and population standard deviation use all eleven
snapshots, counted once each, across the 48 training trajectories and all
spatial cells. Both input and target use this same affine transform:

```text
normalized_u = (u - field_mean) / field_std
physical_prediction = normalized_prediction * field_std + field_mean
```

Each coefficient has a separate mean and population standard deviation
across the 48 training coefficient rows. Statistics do not depend on
repeated coefficient channels or on the evaluation splits. Standard
deviations are floored at `1e-12` for degenerate inputs. Nothing is
normalized per sample, clipped, or logarithmically transformed.

| Quantity | Mean | Standard deviation |
| --- | ---: | ---: |
| Field | 0.12027802015661833 | 0.2355322120279037 |
| cx | -0.12289070488872648 | 0.5691624809559207 |
| cy | -0.019631126517232585 | 0.5642181456221029 |
| nu | 0.031608018258263024 | 0.01183780749459938 |

Validation and test data use these frozen statistics. Checkpoints and
manifests store them so inference never needs to refit a scaler.

## Architecture and Fourier convention

The model is implemented from scratch in PyTorch:

```text
(B,4,64,64)
  -> 1x1 lift, 4 to 32 channels
  -> four blocks: GELU(SpectralConv(v) + Conv1x1(v))
  -> 1x1 projection, 32 to 64 channels, GELU
  -> 1x1 projection, 64 to 1 channel
  -> (B,1,64,64)
```

There are no coordinate channels, padding, dropout, batch normalization,
or output positivity constraints. For a constant-coefficient periodic
PDE, absolute position is not needed: translating the input translates
the correct output. Shared pointwise maps and Fourier convolutions respect
grid translations. A circular-shift test checks this structural property.

Each spectral block uses `rfft2` with `norm='ortho'`, producing
`(B,32,64,33)` complex coefficients. The last axis contains only the
nonnegative y frequencies because the input is real. The x axis still
contains positive and negative frequencies.

The implementation retains x slices `:12` and `-12:` and y slice `:12`.
Thus the precise x indices are 0 through 11 and -12 through -1; y indices
are 0 through 11. Two independent complex weight arrays of shape
`(32,32,12,12)` mix channels at those frequencies. Other output spectral
entries are zero. An `irfft2` with matching normalization and explicit
`s=(N,N)` reconstructs a real field. Invalid overlapping/out-of-range
mode blocks are rejected. Real-FFT boundary components are interpreted
according to PyTorch's inverse real FFT convention.

The weights learn amplitude/phase responses between feature channels;
they are not direct estimates of the PDE coefficients. Coefficient
conditioning enters through the broadcast channels and nonlinear blocks.
The local branch and nonlinearities mean the complete model output is
not strictly restricted to the retained spectral frequencies.

There are **1,186,209 trainable tensor elements** under PyTorch's usual
`sum(p.numel())` convention. Since complex elements contain independently
trainable real and imaginary components, there are **2,365,857 real scalar
parameters**. Both counts are recorded in the manifest to avoid ambiguity.

## Hard overfit gate

Every CLI run first trains a separate model on eight fixed training
pairs: snapshot 0 to 1 from trajectories
`[0,50,21,20,38,10,45,59]`. All eight fit in one batch. This uses Adam at
`1e-3` for 300 updates. Criteria are fixed before the diagnostic:

- Final normalized MSE must be at most 5% of initial MSE.
- Final mean physical relative L2 must be at most 0.05.
- Metrics must be finite.

A failed gate writes its results and exits before full training. It does
not silently relax criteria or try another hyperparameter configuration.
A test explicitly verifies that a failed gate cannot start the full loop.
After success, the gate model is discarded and the seed is reset so full
training begins with fresh initial weights and a fresh optimizer.

The observed gate passed:

| Metric | Before training | After 300 updates |
| --- | ---: | ---: |
| Normalized MSE | 1.3157627582550049 | 0.000008653751137899235 |
| Physical relative L2 | 0.8830530247272522 | 0.0021964207217435873 |
| Relative mass error | 0.23922905282526646 | 0.0004736315023350327 |

Gate optimization/evaluation runtime was 36.51 seconds. This is an
optimization sanity check, not a held-out performance measurement.

## Training and evaluation protocol

Full training uses Adam, learning rate `1e-3`, default Adam betas and eps,
zero weight decay, batch size 8, and 100 epochs. There is no scheduler,
early stopping, hyperparameter search, or physics penalty. The loss is
normalized-space MSE. Validation runs after every epoch, and the checkpoint
with the lowest validation physical relative L2 is retained (first epoch
wins an exact tie). Test data are evaluated only after checkpoint selection.

Python, NumPy, and PyTorch use training seed 2028. Deterministic PyTorch
algorithms are enabled. CPU and the recorded numerical environment are
the reference; bitwise agreement across library versions or hardware is
not promised. The split seed is separate from the training seed and from
the Stage 2 generation seed.

For each pair, predictions and targets are de-normalized and evaluated in
float64. The targets in this evaluation path have already passed through
the Dataset's float32 conversion; consequently there is a small target
quantization floor relative to the original float64 archive. No clipping
or mass correction is applied.

```text
relative_L2 = ||prediction - target||₂ / max(||target||₂, 1e-12)
relative_mass_error = |sum(prediction) - sum(target)|
                      / max(|sum(target)|, 1e-12)
```

Cell area cancels in both ratios on this common uniform grid. Mass error
compares predicted to target mass: it is an evaluation diagnostic, never
a loss or hard conservation constraint. Near-zero target mass would make
this relative diagnostic sensitive, but the development archive contains
positive Gaussian-mixture mass.

Ten pair metrics are averaged within each trajectory, then the trajectory
means are averaged equally. The JSON report preserves every pair and
trajectory metric. The persistence baseline predicts `u(t+0.1)=u(t)` from
the same input. It is essential here because the short prediction horizon
can make copying the field look competitive. Persistence naturally
preserves mass, so low mass error alone does not imply correct transport.

## Measured baseline results

The fixed 100-epoch run completed successfully. Checkpoint selection used
validation only; the selected epoch was **31**, not the final epoch.

| Evaluation | Physical relative L2 | Relative mass error |
| --- | ---: | ---: |
| Selected-checkpoint validation | 0.0192502481 | 0.0109538302 |
| Selected-checkpoint test | 0.0235789974 | 0.014033301 |
| Persistence test | 0.084391747 | 9.68358732e-10 |

Test FNO relative L2 is about 2.358%, versus 8.439% for
persistence. These are means across eight held-out trajectories, each
containing ten one-step pairs. They do not establish rollout stability,
OOD performance, or reliable population-level confidence intervals.

| Test trajectory | FNO relative L2 | FNO relative mass error |
| --- | ---: | ---: |
| 16 | 0.0343706972 | 0.0108953212 |
| 17 | 0.0109515459 | 0.00476581173 |
| 22 | 0.0243504159 | 0.0180668631 |
| 28 | 0.0218932937 | 0.0142615153 |
| 40 | 0.021389353 | 0.0105598251 |
| 47 | 0.0245714084 | 0.0160187851 |
| 49 | 0.0214711969 | 0.00154062295 |
| 51 | 0.0296340677 | 0.0361576632 |

Full training, including per-epoch validation and checkpoint/history I/O,
took **832.93 seconds (13.88 minutes)** on CPU
with four intra-op threads. Total run time after archive loading, including
the gate and final evaluation, was **871.38 seconds**.

The fixed learning rate produced intermittent loss spikes. For example,
validation relative L2 rose to about 0.10953 at epoch 58 before recovering;
final-epoch validation relative L2 was about 0.02550. The epoch-31 checkpoint
remained best. No nonfinite losses, outputs, or gradients were encountered.
No scheduler or other tuning was added in response to test performance.

The FNO's mean test mass discrepancy is about 1.403%,
while persistence has negligible mass discrepancy. The latter is expected:
copying a field preserves its mass even when its position and shape are
wrong. The tiny nonzero persistence discrepancy here includes the float32
normalization/target conversion floor, not just solver roundoff.

## Run artifacts

Each run creates:

- `manifest.json`: status, archive hash, source hashes, Git revision,
  architecture, both parameter counts, split IDs, normalization, seeds,
  package versions, CPU configuration, gate status, selected checkpoint
  epoch/score/hash, and runtimes.
- `overfit.json`: criteria, exact gate pairs, initial/final losses and
  physical errors, per-pair diagnostics, and elapsed time.
- `history.json`: every epoch's training loss, validation metrics, and runtime.
- `best.pt`: selected model state, architecture, normalization, split IDs,
  and training seed. This is an inference checkpoint, not a resume-training
  snapshot with optimizer/RNG state.
- `evaluation.json`: selected-checkpoint validation/test and persistence
  test metrics, including per-pair mass errors.

The Git revision alone does not capture uncommitted Stage 3 implementation;
source hashes record the actual ML package and training-script bytes used.
Archive hashes connect the split IDs to a specific archive. Training
artifacts are local and are not intended for a source-code commit.

## Verification and scope

The complete suite passed: **109 cases** (60 Stage 1, 30 Stage 2, and
19 Stage 3). Hash comparisons also confirmed that every Stage 1/2 package
source file and the existing Stage 2 archive remained byte-for-byte unchanged.

Tests cover trajectory partitioning, pair alignment and coefficient
broadcasting, deterministic shuffling, time-schedule validation,
training-only statistics, affine round trips, finite zero-variance
normalization, FNO shapes/gradients, circular-shift consistency, mode
bounds, parameter counts, physical metrics and aggregation, a training
step/checkpoint round trip, and gate failure behavior. Stage 1 and Stage 2
tests remain unchanged.

The baseline has many parameters relative to 48 training trajectories.
Adjacent pairs are correlated, and the eight-trajectory test set is small.
Numerical labels include finite-difference and RK4 errors; the model can
learn discretization artifacts. Gaussian mixtures are a narrow initial
condition family. Neither mass conservation nor positivity is enforced.
One-step performance does not establish stable long-term evolution.

Autoregressive rollout evaluation, OOD parameter splits, resolution
transfer, GPU benchmarking, and hyperparameter searches are explicitly
outside Stage 3 and have not been implemented.
