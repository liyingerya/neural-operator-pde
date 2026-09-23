# Stage 5: rollout and mass-aware training ablations

This experiment changes training objectives while retaining the Stage 3 FNO
architecture, coefficient channels, trajectory split, and global normalization.
Stage 1/2 are unchanged. The historical epoch-31 checkpoint and every Stage 4
archive/result remain immutable references. There are exactly three reported
models: A (historical), B (rollout-only), C (selected rollout + mass penalty).
The three lambda candidates are validation-only candidates for C, not three
additional claimed model improvements.

## Motivation and hypotheses

Stage 4 measured roughly 2.36% mean ID one-step error but 17.13% final
10-step autoregressive error. Final ID mass error was about 8.19%, and
negative predictions appeared. The hypotheses are that exposure to a model's
own predictions during training can reduce feedback errors, and that a soft
mass penalty can reduce conservation errors. Neither hypothesis guarantees
improvement, stability, positivity, or better OOD behavior.

## Fixed protocol

The protocol is written to `runs/stage5/protocol.json` before any training.
All full runs use the same architecture (width 32, four Fourier blocks,
12 retained modes, four input channels, one output), input/target transforms,
Adam optimizer, and shuffled-window ordering. There are 1,186,209 trainable
tensor elements or 2,365,857 real scalar components accounting for complex
weights. No coordinates, residual architecture, extra data, clipping,
projection, gradient clipping, scheduler, or physics-informed PDE residual is
introduced.

| Setting | Value |
| --- | --- |
| Training trajectories | Original Stage 3 48 |
| Validation trajectories | Original Stage 3 8 |
| Test trajectories | Original Stage 3 8; used only after selection |
| Prediction steps K | 5, equally weighted |
| Training windows | 288 (six per trajectory) |
| Validation windows available | 48; checkpoint selection uses full trajectories instead |
| CPU threads | Four intra-op threads |
| Model initialization seed | 2028 |
| DataLoader shuffle seed | 5029 |
| Batch size | 8 |
| Adam learning rate | 0.0005 |
| Adam betas / epsilon / weight decay | 0.9, 0.999 / 1e-8 / 0 |
| Epoch budget | 20 for B and each C candidate |
| C lambda candidates | 0.01, 0.1, 1.0 |
| Mass denominator floor | 1e-12, in squared physical-mass units |

The initialization reproduces the historical seed/constructor procedure with
the same PyTorch version, never loading trained weights into B or C. The
initial parameter-state hash is recorded and checked identically for all new
runs. Each gate starts anew, and all gate-trained weights are discarded.

The unchanged trajectory IDs, in their saved order, are:

```text
train: [0,50,21,20,38,10,45,59,29,31,35,8,58,46,13,34,30,33,61,44,11,32,7,53,
        26,52,42,37,6,12,36,54,2,23,18,39,15,24,1,63,4,62,57,19,56,27,55,14]
validation: [60,41,5,9,48,25,43,3]
test: [51,16,49,17,40,47,22,28]
```

The public JSON retains the exact saved normalization and initialization hash.
Field mean/std are 0.12027802015661833 / 0.2355322120279037; the coefficient
statistics remain ordered `(cx, cy, nu)`. The reference environment is
Python 3.10.8, NumPy 2.2.6, and PyTorch 2.14.0. Deterministic CPU algorithms,
a seeded DataLoader generator, and zero DataLoader workers are used.

The lower learning rate is fixed up front for full five-step backpropagation,
given the historical fixed-rate training fluctuations. Twenty epochs is a
modest CPU budget: 720 optimizer updates and 28,800 supervised predicted fields
per candidate. These are repeated, correlated observations, not independent
samples. B and C are tightly controlled against each other. Comparisons with A
are historical comparisons, not a fully isolated single-factor experiment:
A had 100 one-step epochs, learning rate 0.001, and a different example/order
structure. Its checkpoint was selected on one-step validation loss, whereas
B/C are selected on final-rollout validation error. No freshly retrained
one-step control is added because the requested
study has exactly three reported variants.

## Windows, feedback, and field objective

For snapshots 0 through 10 and K=5, valid starts are 0 through 5. A window
returns input `(4,N,N)`, targets `(5,1,N,N)`, the float64 physical start mass,
and the trajectory/start indices. Batching adds the batch axis. Windows never
cross trajectory boundaries, and membership is inherited directly from the
frozen Stage 3 split. Normalization is loaded from the baseline manifest and
is never fitted on windows, validation, test, or OOD data.

Only the first field is a true input. Every later input uses the previous
normalized prediction and the same fixed normalized coefficient channels.
Nothing is detached between the five steps; the complete feedback chain is
backpropagated. The normalized-space field objective is

```text
L_field = mean over batch, prediction steps, channels, and spatial cells
          of (predicted_normalized_field - true_normalized_field)^2.
```

All five horizons have equal weight. Targets are float32 training tensors;
physical evaluation uses the unchanged Stage 4 float64-reference path.

## Differentiable mass objective

For each window, the reference is the original float64 start field's physical
mass, `M0 = h² sum(u_start)`, with `h=2π/N`. Predicted physical fields are
formed within the computational graph using the saved affine inverse
normalization. Conversion to float64 for mass accumulation remains
differentiable. No predictions are detached.

```text
M(pred_j) = h² sum(field_std * predicted_normalized_j + field_mean)
L_mass = mean over batch and j of (M(pred_j)-M0)^2 / max(M0^2, 1e-12)
L_total = L_field + lambda_mass * L_mass
```

B has lambda zero; mass is still recorded diagnostically. C has one of the
three predeclared nonzero weights. The tiny denominator floor handles the
formula's zero-mass edge case; this experiment has positive Gaussian-mixture
mass. A soft integral penalty does not enforce local positivity, and matching
mass alone does not establish field accuracy.

## Hard sanity gates

Tests first verify window shapes/membership, true autoregressive feedback,
fixed conditioning, gradients through all K steps, and finite nonzero mass
loss gradients. Every candidate (including B) then attempts to fit eight fixed
windows: starts at t=0 from trajectories `[0,50,21,20,38,10,45,59]`.

Each gate uses the same fresh initialization and optimizer configuration,
up to 300 updates. It is checked every 25 updates and may stop early only when
all predeclared criteria pass:

- normalized field MSE at most 5% of its initial value;
- mean physical relative L2 across the five steps at most 0.05;
- finite field, mass, and physical-error metrics;
- for every C candidate, mass loss at most 10% of its initial value.

If any gate fails, no full training begins. These are optimization sanity
criteria, not required held-out performance. They are not relaxed after seeing
results. The gate report retains initial/final metrics, updates, windows, and
history for each lambda.

## Validation-only selection

After each full epoch, evaluate a ten-step autoregressive rollout from t=0
on the eight existing validation trajectories. The primary checkpoint score
for B and all C candidates is the mean final t=1 physical relative L2. Only
an exact field-error tie is broken by lower final mass error, then earlier
epoch. Across C candidates, smaller lambda breaks any remaining exact tie.
Mass diagnostics and entire validation histories are saved even when field
accuracy selects a less conservative model.

Each C candidate receives the same complete 20-epoch budget and initialization
as B. The best validation checkpoint across these candidates is model C;
there is no extra winner-only refit or additional training budget. This makes
the lambda study small and explicit. The selection file must exist, all gates
must have passed, and checkpoint hashes/metadata must agree before the
separate evaluation command can run. No Stage 4 OOD archive or measured OOD
result is read by the training command. Reading the original archive into
memory does not authorize using its test rows for fitting or selection.

## Frozen Stage 4 evaluation

After validation selection is immutable, A/B/C are evaluated using the
unchanged Stage 4 evaluator and existing archives, verified by hash. No
regeneration, resplitting, coefficient-range expansion, or normalization
refitting occurs. The full evaluation includes original ID test, fresh ID,
high nu, all four high-velocity quadrants and their combined collection, and
the matched 64/128 archives. The historical A rollout is checked numerically
against the published Stage 4 result.

Reports retain teacher-forced one-step errors, all autoregressive horizons,
feedback amplification, physical mass errors/drift/accumulated changes,
negative-cell and negative-mass fractions, and extrema. Aggregation stays at
the trajectory level. Ratios are guarded when teacher-forced errors approach
zero. Both grid resolutions use the same learned weights and statistics;
the 128-grid reference remains a numerical solver solution, not an exact
continuum solution.

## Commands and artifacts

```bash
.venv/bin/python -m pip install -e '.[test,evaluation]'
.venv/bin/python -m pytest -q
.venv/bin/python -m examples.train_stage5
.venv/bin/python -m examples.evaluate_stage5
.venv/bin/python -m examples.report_stage5
```

The default public JSON/CSV paths must be absent for the initial export.
When the published summaries already exist, reproduce into a fresh ignored
directory instead (the figures are regenerated under the selected run):

```bash
.venv/bin/python -m examples.report_stage5 --public runs/stage5/report_copy/stage5_summary.json
```

Use a new `--public` directory for each export; existing summary files are
never overwritten. For an independent full reproduction, train with
`--output runs/stage5_reproduction`, evaluate with
`--run runs/stage5_reproduction`, and report with that same `--run` and a fresh
`--public` path. Frozen Stage 3/4 local artifacts are prerequisites.

Training creates a new output directory and refuses to overwrite a prior
study. Evaluation likewise refuses an existing output directory. Detailed manifests, validation histories, initial/final gate results,
selected checkpoints, predictions, and figures belong under ignored
`runs/stage5/`. Public results and reproducible plotting commands are documented
with the completed study. All older source/data/run files are checked against
pre-implementation hashes. No commit or push is performed during implementation.

The new `stage5_training/` package separates window construction (`data.py`),
the differentiable objectives (`objective.py`), fixed configuration (`config.py`),
training/validation/gates (`training.py`), checkpoint verification
(`checkpoint.py`), and public summaries/plots (`reporting.py`). The three
`examples/*stage5.py` commands separate training, frozen evaluation, and
reporting. Stage 5 tests are in `tests/test_stage5_*.py`. Only README and
package discovery in `pyproject.toml` need changes outside these additions.

## Observed tiny-window gates

All gates passed before any full run began.

| Lambda | Updates | Field MSE, initial → final | Mass loss, initial → final | Physical L2, initial → final | Seconds |
| --- | ---: | --- | --- | --- | ---: |
| 0 | 200 | 1.28187 → 0.00294944 | 0.0806138 → 0.00642302 | 88.069% → 4.185% | 121.66 |
| 0.01 | 200 | 1.28187 → 0.00281031 | 0.0806138 → 0.0058832 | 88.069% → 4.081% | 129.38 |
| 0.1 | 175 | 1.28187 → 0.00315446 | 0.0806138 → 0.00533731 | 88.069% → 4.292% | 106.17 |
| 1 | 150 | 1.28187 → 0.00289928 | 0.0806138 → 0.00110904 | 88.069% → 4.026% | 95.64 |

## Limits of interpretation

This is one seed on 48 training and eight validation trajectories. The three
lambda candidates create some validation-selection optimism; there are no
confidence claims or seed-averaged conclusions. Five-step training supervises
half the final evaluation horizon, so ten-step stability is still an empirical
question. Relative mass penalties weight errors relative to each trajectory's
mass and can compete with spatial accuracy. Cancellation of positive and
negative errors can produce a small integral error despite a poor field.

The 64/128 experiment transfers the same weights and normalization without
retraining. Matched physical initial conditions do not make either discrete
solver reference exact, and the native-grid error difference includes changes
in the numerical reference. CPU wall times include training and per-epoch
validation and depend on host load; they are not a new inference speed benchmark.
There is no new OOD training, rollout correction, positivity guarantee, PDE
residual loss, or architecture change.

## Validation selection and training runtimes

All full runs completed their predeclared 20 epochs. The following values are
at each candidate's best **validation** checkpoint, selected without test/OOD data.

| Candidate | Lambda | Selected epoch | Validation final AR L2 | Validation mass error | Training + validation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | 0 | 16 | 21.197% | 12.653% | 468.73 |
| C candidate | 0.01 | 16 | 20.488% | 13.158% | 455.08 |
| C candidate | 0.1 | 16 | 22.385% | 14.315% | 443.27 |
| C candidate | 1 | 16 | 28.272% | 15.515% | 446.21 |

C uses lambda **0.01**. At these field-selected checkpoints, none of the mass
candidates improves validation mass error over B. Selecting a checkpoint for
its lowest mass error instead would violate the predeclared primary criterion.
This negative validation finding is retained; no additional tuning is performed.

The selected checkpoint SHA-256 values are:

```text
A epoch 31: 300f7c54920c3d5d867fca1e059f7cd1024bc46499fc3c2a15a4b63c8b322f87
B epoch 16: b28df728a667aacb511cdd78357c05c6b8d745e010cdd10ee1e54fe81131554c
C epoch 16: e0444d0fbd29befccacad0931c5fbd528eadb3bbdb669f9d01a3ff9b6985006b
```

B is saved at `runs/stage5/rollout/best.pt`; C at
`runs/stage5/mass_0.01/best.pt`. Full candidate runs took
1813.29 seconds in total.
The four gates took another 452.86 seconds. These are CPU wall times with four
intra-op threads on an Apple M1; no GPU training or benchmarking was used.

## Held-out results

**This fixed-budget experiment does not improve on historical A.** B and C
have larger one-step errors, final rollout errors, mass violations, and ID
negative-mass fractions than A. C makes small improvements over B in several
aggregate measures, but the selected soft penalty does not recover A's quality.
The modest B/C schedule, changed checkpoint criterion, and single seed limit
causal comparisons with A. These results do not establish that rollout training
or mass penalties cannot help under a different, independently designed study.

Errors below are physical relative errors, averaged over trajectories. One-step
ID L2 averages the ten teacher-forced transitions; final errors are at t=1.
Public [JSON](results/stage5_summary.json) retains all ten evaluation regimes,
every horizon, feedback ratios, all autoregressive physical diagnostics,
failures, selection provenance, and the complete candidate validation comparison.
The compact [CSV](results/stage5_ablation.csv) contains the primary table.
Detailed histories and trajectory-level predictions remain ignored.

| Model | One-step ID L2 | Final ID AR L2 | Final ID mass error | High-velocity final AR L2 | Native 64 AR L2 | Transferred 128 AR L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 2.358% | 17.134% | 8.188% | 40.581% | 14.837% | 14.947% |
| B | 6.352% | 28.848% | 25.676% | 55.181% | 24.287% | 24.347% |
| C | 6.381% | 28.215% | 24.705% | 53.687% | 24.290% | 24.355% |

The frozen persistence baseline has final ID error 76.550%. All three FNOs
beat persistence on this ID aggregate, but B/C do not beat the historical FNO.

### ID errors at every prediction horizon

| Time | A TF L2 | B TF L2 | C TF L2 | A AR L2 | B AR L2 | C AR L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 2.520% | 6.214% | 6.215% | 2.520% | 6.214% | 6.215% |
| 0.2 | 2.474% | 6.201% | 6.205% | 4.557% | 9.761% | 9.663% |
| 0.3 | 2.431% | 6.203% | 6.212% | 6.431% | 12.850% | 12.699% |
| 0.4 | 2.391% | 6.222% | 6.238% | 8.197% | 15.642% | 15.446% |
| 0.5 | 2.356% | 6.260% | 6.283% | 9.874% | 18.218% | 17.968% |
| 0.6 | 2.325% | 6.315% | 6.346% | 11.470% | 20.617% | 20.301% |
| 0.7 | 2.298% | 6.387% | 6.427% | 12.990% | 22.863% | 22.473% |
| 0.8 | 2.276% | 6.473% | 6.520% | 14.438% | 24.977% | 24.505% |
| 0.9 | 2.259% | 6.571% | 6.625% | 15.818% | 26.969% | 26.415% |
| 1.0 | 2.248% | 6.678% | 6.739% | 17.134% | 28.848% | 28.215% |

| Time | A feedback ratio | B feedback ratio | C feedback ratio |
| --- | ---: | ---: | ---: |
| 0.1 | 1.000x | 1.000x | 1.000x |
| 0.2 | 1.842x | 1.574x | 1.557x |
| 0.3 | 2.646x | 2.072x | 2.044x |
| 0.4 | 3.428x | 2.514x | 2.476x |
| 0.5 | 4.191x | 2.910x | 2.860x |
| 0.6 | 4.933x | 3.265x | 3.199x |
| 0.7 | 5.652x | 3.579x | 3.497x |
| 0.8 | 6.343x | 3.858x | 3.758x |
| 0.9 | 7.001x | 4.104x | 3.987x |
| 1.0 | 7.622x | 4.320x | 4.187x |

Feedback is the ratio of trajectory-mean AR error to trajectory-mean
TF error at that horizon. Lower ratios for B/C are **not** evidence of better
absolute rollout accuracy: their teacher-forced denominators are much larger.
The t=0 ratio is undefined and stored as null.

### Final ID physical diagnostics

| Diagnostic | A | B | C |
| --- | ---: | ---: | ---: |
| Target mass error | 8.188% | 25.676% | 24.705% |
| Signed mass drift from u0 | -1.836% | -13.435% | -13.542% |
| Accumulated absolute mass changes | 9.108% | 29.143% | 27.678% |
| Negative-cell fraction | 5.914% | 23.813% | 27.161% |
| Negative cells below tolerance | 5.911% | 23.813% | 27.161% |
| Negative-mass fraction | 0.069% | 1.084% | 1.448% |
| Mean trajectory minimum | -0.002349 | -0.012459 | -0.014062 |
| Mean trajectory maximum | 1.046047 | 0.967353 | 0.960910 |
| Global minimum across test trajectories | -0.004563 | -0.020516 | -0.022588 |
| Global maximum across test trajectories | 1.386540 | 1.380511 | 1.340946 |

Signed mass drift can cancel across trajectories; absolute mass error is
reported separately. Accumulated changes sum absolute step-to-step mass
changes and divide by initial absolute net mass. Negative mass uses the
initial absolute-field integral as denominator. The numerical tolerance is
`1e-7 * max(1, max(abs(u0)))`, as in Stage 4. A soft mass penalty does not
prevent negative values: C slightly reduces ID mass error versus B while
increasing its negative-cell and negative-mass fractions.

### Fresh-ID and OOD results

All archive definitions, seeds, and ranges remain those of
[Stage 4](stage4_evaluation.md); there is no new OOD tuning. Quadrant labels
give the signs of `(cx,cy)`. Values are final autoregressive relative L2.

| Distribution | A | B | C |
| --- | ---: | ---: | ---: |
| fresh_id | 15.412% | 23.953% | 24.276% |
| high_nu | 14.059% | 24.643% | 24.458% |
| high_velocity | 40.581% | 55.181% | 53.687% |
| velocity_pp | 36.782% | 63.733% | 62.077% |
| velocity_pn | 50.110% | 50.993% | 51.050% |
| velocity_np | 34.146% | 40.669% | 39.561% |
| velocity_nn | 41.286% | 65.329% | 62.059% |

Final relative target-mass errors for the same distributions:

| Distribution | A | B | C |
| --- | ---: | ---: | ---: |
| fresh_id | 8.750% | 19.865% | 18.592% |
| high_nu | 6.103% | 18.435% | 18.238% |
| high_velocity | 13.196% | 39.191% | 37.990% |
| velocity_pp | 11.181% | 58.970% | 55.413% |
| velocity_pn | 17.882% | 23.489% | 25.488% |
| velocity_np | 9.199% | 14.164% | 15.497% |
| velocity_nn | 14.524% | 60.140% | 55.562% |

High-velocity final negative-cell fractions are 16.285%, 48.893%, 44.015% for A/B/C;
negative-mass fractions are 0.742%, 4.696%, 3.929%.
C improves combined high-velocity field error over B but not over A.
Its negative-negative quadrant remains especially problematic physically.
Fresh-ID accuracy is slightly worse for C than B, so the improvement is not
uniform across evaluation distributions.

### Matched 64/128 transfer

| Model | Native 64 final AR L2 | Native 128 final AR L2 | Difference, percentage points | Shared-grid prediction relative L2 |
| --- | ---: | ---: | ---: | ---: |
| A | 14.837% | 14.947% | 0.1096 | 1.138e-07 |
| B | 24.287% | 24.347% | 0.0601 | 1.282e-07 |
| C | 24.290% | 24.355% | 0.0646 | 1.294e-07 |

The comparison samples the 128 prediction on shared 64-grid points and uses
the sampled 128 prediction norm as denominator. AR predictions agree closely
across grids, but that agreement does not establish continuum accuracy.
The reference at 128 remains a discretized numerical solver solution.
No model is retrained, and no normalization is refitted for either resolution.

### Failures, figures, and runtime

No nonfinite rollout failures occurred in any model/distribution. Nevertheless,
negative values, mass drift, and substantial finite field errors remain.
Trajectory 16 is the worst final ID case for all three models: 27.015%, 40.775%, 40.325%.
The field plots show displaced and damped peaks, particularly for B/C.

`python -m examples.report_stage5` reproduces six ignored PNG figures under
`runs/stage5/figures/`: `id_rollout.png`, `id_mass.png`, `high_velocity.png`,
`representative.png` (first saved test ID, 51), `baseline_worst_id.png`
(baseline-defined worst ID, 16, held fixed across A/B/C), and
`lambda_tradeoff.png`. Field and error panels use shared scales across models.
All six figures were visually inspected for labels, scales, and clipping.

Evaluation wall times (including saved predictions) were 49.41s, 49.36s, 49.88s for A/B/C.
The only runtime notices were Matplotlib building its local font cache and
macOS reporting its static font registry. No numerical training warnings or
nonfinite-gradient failures occurred.

## Regression and protected-file verification

The complete Stage 1–5 suite reports **150 passed in 2.57 seconds**: 131
existing tests plus 19 Stage 5 tests. New checks cover window membership and
alignment, unchanged normalization, fixed coefficients, full K-step feedback
gradients, differentiable physical mass, equal horizon weighting, lambda
restrictions, deterministic initialization/shuffling, the hard-gate failure
path, validation selection, checkpoint provenance, and compatibility with the
unchanged Stage 4 evaluator.

All **98 protected files/artifacts** match SHA-256 snapshots taken before
implementation. These include Stage 1–4 source and tests, earlier example
commands and public results, the Stage 2 archive, historical Stage 3 runs and
checkpoint, and Stage 4 datasets/results/figures/hardware records. Frozen
training-source hashes and the public summary's Stage 5 source hashes also
match. The verification record is ignored at
`runs/stage5/protected_verification.json`.

All 89 generated Stage 5 files present at verification are ignored through
the existing `/runs/` rule; `/data/` remains ignored. Both personal learning
notes retain their repository-local exclude rules. `git diff --check` passes.
Only README and `pyproject.toml` are modified tracked files; the new Stage 5
package, three commands, four test files, this document, and two public summary
files are untracked additions awaiting review. Nothing was staged, committed,
or pushed.
