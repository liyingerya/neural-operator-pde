# Stage 4: frozen FNO evaluation beyond one-step prediction

Stage 4 evaluates the approved epoch-31 Stage 3 checkpoint without training,
normalization fitting, clipping, mass projection, positivity corrections, or
architecture changes. Stage 1/2/3 source behavior, the original development
archive, and the complete Stage 3 run remain unchanged.

The main findings are substantial feedback error accumulation and mass drift,
poor high-velocity extrapolation, and no CPU speed advantage at this problem
size. The matched 128×128 evaluation has similar native-grid error to 64×64.
These observations characterize this baseline; they are not acceptance
thresholds or a claim of general scientific reliability.

## Reproduce the evaluation

Install the optional evaluation dependencies and run from the repository root:

```bash
.venv/bin/python -m pip install -e '.[test,evaluation]'
.venv/bin/python -m examples.evaluate_stage4 --phase id
.venv/bin/python -m examples.generate_stage4_data
.venv/bin/python -m examples.evaluate_stage4 --phase ood
.venv/bin/python -m examples.benchmark_stage4
.venv/bin/python -m examples.generate_stage4_data --resolution
.venv/bin/python -m examples.evaluate_stage4 --phase resolution
.venv/bin/python -m examples.plot_stage4
.venv/bin/python -m examples.summarize_stage4
.venv/bin/python -m pytest -q
```

The original `data/dev/trajectories.npz` and `runs/stage3_baseline/` must be
available locally. They are intentionally not distributed in Git. No Stage 4
command retrains a missing checkpoint: loading fails instead. The loader
checks the archive hash, recorded Stage 3 source hashes, checkpoint hash,
epoch, architecture, split IDs, and normalization consistency. Historical
absolute training-script paths in the manifest are resolved to this checkout.

The frozen checkpoint SHA-256 is:

```text
300f7c54920c3d5d867fca1e059f7cd1024bc46499fc3c2a15a4b63c8b322f87
```

This checkpoint was trained before the Stage 3 source commit; its historical
manifest revision names the earlier Stage 2 commit. Source hashes identify the
actual Stage 3 implementation, and Stage 4 verifies those bytes. The approved
Stage 3 source commit is `9f1cc21d5187c50c4658f5cb36cae0b14dd5f1b1`.

JSON results and generated archives refuse silent replacement. Use fresh
output paths when re-running; default benchmark/plot/summary scripts expect
the default data/run directories. The checked-in summary can be regenerated
to another filename with `examples.summarize_stage4 --output PATH`. Plotting
can redraw existing figures without recomputing inference. Generated files
are under the existing ignored `/data/` and `/runs/` directories. Only the
small public JSON/CSV summaries and this document are intended for Git.

On this machine the sandbox denied detailed hardware queries. A read-only
`/usr/sbin/sysctl -n machdep.cpu.brand_string hw.physicalcpu hw.logicalcpu`
query outside the sandbox confirmed Apple M1, 8 physical and 8 logical cores.
That observation and NumPy build details are recorded separately in the local
`hardware.json`. Missing hardware details remain explicitly unavailable on
other hosts rather than being guessed.

## Methods and aggregation

The original eight Stage 3 test trajectory IDs remain
`[51,16,49,17,40,47,22,28]`. Training/validation membership is not redefined.
All new archives are evaluation-only.

At each horizon `t_k`, teacher forcing supplies the true field at `t_(k-1)`
to the frozen FNO. Autoregression supplies its own prior prediction instead.
The normalized output can be fed back directly because input and output use
the same saved global field scaler. Coefficients are normalized once and
held fixed as three constant channels throughout feedback. Persistence
rollout retains `u0` at every horizon; it does not reset to each true field.

Each model call has input `(B,4,N,N)` and output `(B,1,N,N)`. Saved physical
trajectories have shape `(M,11,N,N)`, with the true initial field at index zero.
The step is always 0.1; incompatible time schedules are rejected. The model
runs in evaluation/inference mode with parameter gradients disabled. There
are no optimizer steps or calls to `Normalization.fit`.

Predictions are de-normalized and all diagnostics use float64 accumulation
against original float64 archive targets. The FNO computations remain
float32/complex64. This differs slightly from Stage 3's de-normalized float32
targets: its average one-step test error is reproduced to about 7e-10 absolute,
while the persistence mass diagnostic no longer includes that target-casting
floor. The historical Stage 3 results have not been rewritten.

Every horizon summary averages one value per trajectory. A mean across time
first averages horizons within each trajectory. Descriptive standard deviation
and minimum/maximum are stored alongside counts; they are not confidence
intervals based on independent snapshots. The original test set has only eight
independent trajectory realizations. A nonfinite trajectory is marked from its
first failure onward; the aggregate is unavailable if any trajectory fails,
with finite counts retained. No failures are silently removed.

## Physical diagnostics and feedback amplification

For the uniform periodic grid, mass is `M(u)=h² sum(u)`, where `h=2π/N`.
The cell area cancels for within-grid relative comparisons but matters when
reporting physical mass across grids. Epsilon is `1e-12` in norm/mass ratios.

- Relative L2: `||prediction-target||₂ / max(||target||₂, epsilon)`.
- Target mass error: `|M(prediction)-M(target)| / max(|M(target)|, epsilon)`.
- Signed drift: `(M(prediction)-M(u0)) / max(|M(u0)|, epsilon)`.
- Absolute initial drift: the magnitude of signed drift.
- Accumulated absolute changes: sum of absolute successive mass increments,
  divided by initial absolute mass. This prevents cancellation across time.
- Minimum and maximum field value, at every horizon and for every trajectory.
- Negative fraction: proportion of grid entries below zero.
- Negative mass fraction: `h² sum(max(-prediction,0)) / (h² sum(abs(u0)))`.
- A second negative count uses threshold `-1e-7*max(1,max(abs(u0)))` per
  trajectory. It distinguishes strict sign violations from tiny undershoots;
  it never changes predictions.

Feedback amplification is `AR_relative_L2 / TF_relative_L2` only when the
teacher-forced error exceeds `1e-10` and both errors are finite. At time zero
it is undefined, represented by JSON null. Both the mean of per-trajectory
ratios and the ratio of mean errors are saved. Tables/plots below use the
**ratio of mean errors**, which is not generally the mean of ratios. This
comparison is descriptive, not a stability bound or success criterion.

## Evaluation distributions

All cases use eleven snapshots from 0 through 1 and retain Stage 2's Gaussian
width, amplitude, center, and 1–3-blob distributions. No Fourier-family OOD is
introduced. Each OOD archive retains its own accurate Stage 2 metadata.

| Archive | Count | Seed | Changed configuration |
| --- | ---: | ---: | --- |
| Fresh ID | 64 | 4100 | None |
| High diffusivity | 64 | 4101 | nu in [0.055,0.08]; velocities stay in [-1,1] |
| Velocity ++ | 16 | 4110 | cx,cy in [1.05,1.4] |
| Velocity +- | 16 | 4111 | cx in [1.05,1.4], cy in [-1.4,-1.05] |
| Velocity -+ | 16 | 4112 | cx in [-1.4,-1.05], cy in [1.05,1.4] |
| Velocity -- | 16 | 4113 | cx,cy in [-1.4,-1.05] |
| Matched resolution 64 | 32 | 4120 | Original distribution, N=64 |
| Matched resolution 128 | 32 | 4120 | Original distribution, N=128 |

Velocity quadrants are a balanced sign-stratified mixture; both component
magnitudes are outside training support. Diffusivity retains [0.005,0.05].
Combined IDs include the archive name, avoiding collisions among archive-local
IDs. The disconnected velocity distribution is not misrepresented as one
continuous Stage 2 configuration range.

Fresh ID measures generalization within the original distribution. High nu
and velocity measure parameter extrapolation. Different regime seeds also
produce different initial conditions, so differences are not isolated causal
effects of nu or velocity sign. The 16-trajectory quadrant samples are small.

Every generated archive is checked for finiteness, extrema, mass drift, and
internal step counts. The first four trajectories per archive also undergo a
one-interval half-timestep comparison using the unchanged Stage 1 API. These
checks do not establish continuum accuracy or bound all spatial errors.

## Measured results

### ID horizons

All field errors below are percentages; feedback is dimensionless.

| Time | Teacher-forced L2 (%) | Autoregressive L2 (%) | Persistence L2 (%) | Feedback ratio | AR mass error (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 |
| 0.1 | 2.5204 | 2.5204 | 8.6790 | 1.0000 | 1.2421 |
| 0.2 | 2.4737 | 4.5574 | 17.2809 | 1.8423 | 2.2787 |
| 0.3 | 2.4307 | 6.4307 | 25.7380 | 2.6456 | 3.1571 |
| 0.4 | 2.3915 | 8.1969 | 33.9875 | 3.4276 | 3.9046 |
| 0.5 | 2.3561 | 9.8738 | 41.9730 | 4.1908 | 4.6097 |
| 0.6 | 2.3249 | 11.4697 | 49.6462 | 4.9335 | 5.4203 |
| 0.7 | 2.2981 | 12.9896 | 56.9669 | 5.6523 | 6.1733 |
| 0.8 | 2.2762 | 14.4377 | 63.9042 | 6.3429 | 6.8731 |
| 0.9 | 2.2595 | 15.8180 | 70.4362 | 7.0008 | 7.5239 |
| 1.0 | 2.2480 | 17.1340 | 76.5499 | 7.6218 | 8.1881 |

At time zero no model prediction has occurred. The final teacher-forced
error is for the step from the true t=0.9 field to t=1; it is not a ten-step
forecast. Mean one-step error across all ten horizons is 2.35790%, while
final autoregressive error reaches 17.1340%. The worst ID trajectory is 16,
with final error 27.0153%. The predetermined field example is trajectory 51.

### Fresh ID and OOD

| Regime | Count | TF mean across horizons (%) | TF final (%) | AR final (%) | Persistence final (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| id_test | 8 | 2.3579 | 2.2480 | 17.1340 | 76.5499 |
| fresh_id | 64 | 2.2315 | 2.1263 | 15.4120 | 79.9360 |
| high_nu | 64 | 2.1954 | 2.2377 | 14.0594 | 78.2858 |
| high_velocity | 64 | 5.3022 | 4.9578 | 40.5810 | 127.3810 |
| velocity_pp | 16 | 5.0638 | 4.7825 | 36.7821 | 128.5937 |
| velocity_pn | 16 | 6.1211 | 5.8041 | 50.1103 | 126.8950 |
| velocity_np | 16 | 4.6646 | 4.4176 | 34.1456 | 127.5401 |
| velocity_nn | 16 | 5.3592 | 4.8272 | 41.2860 | 126.4950 |

High diffusivity is not harder than fresh ID in this sample. More diffusion
can smooth structures, but independent initial-condition samples prevent a
clean causal attribution. High velocity is clearly weaker here: the +-
quadrant has the largest mean rollout error, and trajectory `velocity_pn:10`
reaches 77.3317% final error. This does not prove a general sign asymmetry.

### Mass and negative fields

The following are trajectory means at t=1; the minimum column is the
most negative final grid value across the ensemble, not a mean of minima.

| Regime | Absolute mass error (%) | Signed initial drift (%) | Accumulated absolute changes (%) | Negative cells (%) | Negative mass fraction (%) | Minimum final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| id_test | 8.1881 | -1.8358 | 9.1083 | 5.9143 | 0.0686 | -0.004563 |
| fresh_id | 8.7500 | -2.8903 | 9.8536 | 20.3228 | 0.6742 | -0.013094 |
| high_nu | 6.1031 | -2.7574 | 9.1539 | 9.7534 | 0.1888 | -0.009879 |
| high_velocity | 13.1964 | -9.8995 | 14.3531 | 16.2853 | 0.7423 | -0.013355 |

Signed drift averaged across trajectories can hide cancellation. For ID,
mean signed drift is -1.8358% while mean absolute mass error is 8.1881%.
The ID final global maximum is 1.38654; the most negative ID value across
all horizons is -0.00898952. Fresh-ID negative cells are common in near-zero
regions, but their integrated negative mass is much smaller than the cell
fraction. The thresholded counts remain almost identical to raw counts,
so these sign violations are not solely below the reporting tolerance.

All predicted trajectories stayed finite, but positivity and conservation
failed to hold. No correction was applied. Persistence maintains initial
mass to solver roundoff while giving large field errors, demonstrating why
mass alone is not an accuracy measure.

### CPU benchmark

Five warm-ups and thirty timed groups were used per method/workload.
A pilot selects repetitions targeting at least 20 ms per group; each group
is divided by its repetition count. Medians and IQRs are retained. Timings
exclude model/archive loading and metrics. Forward-only inputs are already
prepared. End-to-end inference includes normalization, broadcasting, model
execution, and de-normalization into physical arrays.

Batch 1 uses the first listed trajectory and batch 8 uses the first eight;
ID uses the fixed test-ID order. Other regimes use archive IDs 0 onward.
The solver loops sequentially over identical physical problems; FNO uses a
batch. Both advance 0.1, with each solver problem using its generated
base_dt and interval-end shortening. This is an implementation comparison
with different arithmetic precision and accuracy, not equal-accuracy or
fully optimized parallel algorithms.

| Workload | Batch | Solver latency (ms) | FNO forward (ms) | FNO end-to-end (ms) | Solver / FNO | Solver throughput (problems/s) | FNO throughput (problems/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id_test | 1 | 0.345 | 8.165 | 8.129 | 0.0424× | 2899.6 | 123.0 |
| id_test | 8 | 3.778 | 40.427 | 39.795 | 0.0949× | 2117.3 | 201.0 |
| fresh_id | 8 | 3.931 | 39.710 | 40.367 | 0.0974× | 2035.0 | 198.2 |
| high_nu | 8 | 9.715 | 40.171 | 40.232 | 0.2415× | 823.5 | 198.8 |
| velocity_pp | 8 | 5.138 | 41.023 | 40.531 | 0.1268× | 1557.0 | 197.4 |
| velocity_pn | 8 | 5.012 | 39.623 | 40.021 | 0.1252× | 1596.3 | 199.9 |
| velocity_np | 8 | 4.629 | 40.245 | 42.210 | 0.1097× | 1728.4 | 189.5 |
| velocity_nn | 8 | 4.795 | 42.666 | 41.775 | 0.1148× | 1668.3 | 191.5 |

A speedup below one means FNO is slower. ID end-to-end FNO is about
23.57× slower for batch 1 and 10.53× slower for batch 8. This small, smooth,
short-interval NumPy problem needs only a few RK4 updates; a million-element
FNO is not automatically cheaper. High nu increases solver cost, but FNO
still loses here. Forward-only and end-to-end measurements are independent;
small reversals in their measured ordering are timing noise, not negative
preprocessing cost. No GPU/MPS benchmark or performance tuning was done.

### Paired 64 to 128 resolution transfer

The two 32-trajectory archives have identical coefficients, blob parameters,
and snapshot times. Initial fields agree exactly at shared grid points.
The same weights, 12 modes, global statistics, physical domain, and 0.1
prediction horizon are used without fitting. Dynamic FFT sizes and pointwise
convolutions make the existing model technically valid at 128×128.

| Grid | TF mean across horizons (%) | TF final (%) | AR final (%) | AR mass error (%) |
| --- | ---: | ---: | ---: | ---: |
| resolution_64 | 2.0683 | 2.0459 | 14.8370 | 7.5615 |
| resolution_128 | 2.0777 | 2.0487 | 14.9466 | 7.5615 |

Mean final native-grid AR error increases by 0.10961 percentage points.
Per-trajectory differences range from -0.38610 to +0.55183 percentage points.
The two solver references differ by 0.49082% mean relative L2 at shared
points at t=1, using the sampled fine-grid reference as denominator. Their
different spatial/time discretization errors therefore matter.

Autoregressive FNO predictions themselves differ by only about 1.14e-7
relative L2 at shared points at t=1. Teacher-forced predictions differ more
(about 0.44553%) because their solver-provided inputs differ across grids.
Similar native-grid errors demonstrate this specific transfer experiment,
not a proof of resolution-invariant continuum accuracy. **The 128×128
reference is still a discretized solver solution, not an exact solution.**

### Reference checks, runtime, and tests

Maximum absolute mass drift across the newly generated solver archives is
3.55271e-15. The largest first-interval base-vs-half-timestep
relative discrepancy in the tested subset is 2.86879e-06. All generated references
are finite. These checks concern time stepping and conservation; they do
not remove spatial discretization error.

Execution used Apple M1 (8 physical/logical cores), macOS 14.7.4 ARM64,
Python 3.10.8, NumPy 2.2.6, and PyTorch 2.14.0. PyTorch used four intra-op
and eight inter-op threads; no BLAS/OpenMP thread environment overrides
were set. FNO arithmetic was float32/complex64 and the solver float64.

| Timed phase | Seconds |
| --- | ---: |
| id | 1.165 |
| ood | 26.686 |
| resolution | 20.461 |
| benchmark | 42.136 |
| generation | 7.233 |

The recorded phase sum is **97.681 seconds**. It excludes
dependency installation, development, tests, plotting, and loading outside
the timers; it is not a claim about total interactive task duration.

The complete suite passed **131 tests**: the unchanged 109 Stage 1–3
cases plus 22 Stage 4 cases. New coverage includes frozen provenance,
no normalization refit, feedback versus teacher-forcing inputs, fixed
coefficients, time alignment, original target precision, both grid sizes,
nonfinite failure visibility, OOD supports and determinism, matched physical
problems, mass/negative metrics, ratio guards, and finite timing outputs.

One initial test assertion incorrectly expected combined archives to retain
the original valid-trajectory count; the assertion was corrected to require
the doubled count and unchanged means. No inference change was required.
Protected-file hashes verify all Stage 1/2/3 packages, the Stage 3 training
script, the original archive, and all original Stage 3 run artifacts.


## Files and artifacts

| Location | Responsibility |
| --- | --- |
| `stage4_evaluation/checkpoint.py` | Approved checkpoint hash, source/archive verification, frozen loading |
| `stage4_evaluation/rollout.py` | Teacher-forced, autoregressive, and persistence evaluation; collection aggregation |
| `stage4_evaluation/diagnostics.py` | Physical-unit metrics, ratio guards, descriptive summaries |
| `stage4_evaluation/ood.py` | Existing public generator configurations, numerical checks, matched-grid checks |
| `stage4_evaluation/benchmark.py` | CPU workloads, warm-ups, repeated timings, device context |
| `stage4_evaluation/plotting.py` | Figures from saved results, independent of evaluation |
| `examples/*stage4*.py` | Generation, evaluation, benchmarking, plotting, public-summary commands |
| `tests/test_stage4_*.py` | Stage 4 verification |
| `docs/results/stage4_summary.json` | Public horizon summaries, configurations/hashes, timing statistics, comparison results |
| `docs/results/stage4_final_by_trajectory.csv` | Public final-horizon metrics, one row per physical realization/grid |
| `data/stage4/` | Ignored generated archives and manifests |
| `runs/stage4/` | Ignored detailed metrics, predictions, hardware/timing records, figures, and font cache |

The combined high-velocity collection is not double-counted in the public
CSV: its compound trajectory IDs retain quadrant names. Paired grids have
separate rows because their references and native-grid errors differ. The
JSON contains both quadrant breakdowns and the combined collection.

Eight inspected figures are generated: ID error trajectories, feedback
amplification, physical diagnostics, ID/OOD comparison, CPU timing, resolution
transfer, predetermined ID field snapshots, and worst-final-ID snapshots.
Field plots transpose the stored x/y array axes so x is horizontal; truth and
prediction share color limits, and difference plots use symmetric limits.
Generated plots remain ignored and can be reproduced locally.

## Interpretation and limitations

- Good one-step accuracy does not imply good feedback behavior. The ID final
  error is about 7.62 times the teacher-forced final-horizon error, and the
  high-velocity rollout reaches about 40.58% mean error despite a roughly
  5.30% average one-step error. The ratio is not a Lipschitz constant or a
  proof of dynamical instability.
- The model is finite over the tested ten steps but violates mass conservation
  and positivity. Mean signed mass drift can mask opposing errors among
  trajectories. Teacher-forced outputs do not form one physical evolution;
  cumulative mass-change interpretation is primarily for autoregression.
- High-nu results are not evidence that all extrapolation is safe, and velocity
  quadrant differences do not establish sign-specific causes with independent
  small samples. The initial-condition family remains narrow.
- Similar 64/128 errors concern matched smooth Gaussian problems on the same
  physical domain. Reference discretization differences, aliasing, and other
  initial-condition spectra can change the outcome. No continuum limit or
  general resolution-transfer guarantee is established.
- The CPU FNO is slower than the current numerical solver for these small
  workloads. This result must not be generalized to other hardware, PDE
  complexity, batch sizes, or error tolerances. GPU/MPS timing was not run.
- Numerical timestep refinement checked only four initial intervals per new
  archive. It is a sanity check, not a full convergence study. The experiment
  ends at t=1 and does not test very long-time dynamics.

These weaknesses are recorded for Stage 5. No model improvement, new loss,
correction, architecture, retraining, or parameter search is part of Stage 4.
