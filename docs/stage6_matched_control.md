# Stage 6: matched one-step control

Stage 5's negative result did not isolate the rollout objective: historical A
used 100 one-step epochs at learning rate 0.001 and one-step validation selection,
whereas B/C used 20 five-step epochs at 0.0005 and final-rollout selection.
Stage 6 adds D, a one-step model using the frozen Stage 5 initialization,
optimizer settings, and checkpoint-selection rule. A/B/C are not retrained.

## Predeclared protocol

D uses the unchanged Stage 3 FNO (12 modes, width 32, four blocks, projection
width 64), coefficient conditioning, exact saved 48/8/8 trajectory IDs, and
saved normalization. No statistics are fitted. Adam uses learning rate 0.0005,
betas (0.9,0.999), epsilon 1e-8, weight decay zero, and batch size eight.
Initialization seed is 2028; DataLoader shuffle seed is 5029, with zero workers,
no dropped batches, deterministic algorithms, and four CPU threads. The model
has 1,186,209 trainable tensor elements (2,365,857 real scalar components when
complex Fourier parameters are counted as two real components).

The only objective is original one-step normalized-space MSE. There are no
rollout losses, mass penalties, clipping, projections, schedulers, new data,
architecture changes, or searches. All 480 consecutive training pairs are
used each epoch for 60 epochs. Gate-trained parameters are discarded.

| Quantity | B/C, each full run | D, full run |
| --- | ---: | ---: |
| Epochs | 20 | 60 |
| Examples per epoch | 288 windows | 480 pairs |
| Supervised predictions per example | 5 | 1 |
| Optimizer updates | 720 | 3600 |
| Supervised field comparisons | 28,800 | 28,800 |

Matching this count is not a perfect compute match. B/C make five sequential
FNO calls per window and backpropagate through the feedback chain. D updates
weights five times as often over the full run. Exact FLOPs, memory use,
gradient paths, and validation overhead differ. The fields are repeated and
correlated, not independent observations. B/C's overlapping windows give
snapshot targets 1–10 multiplicities `[1,2,3,4,5,5,4,3,2,1]` per epoch; D
weights all ten transitions equally. This remains a training-setup difference.

After every epoch, the unchanged Stage 4 rollout code evaluates the eight
saved validation trajectories. The Stage 5 ordering selects lowest mean final
t=1 physical autoregressive L2, then lower final mass error, then earlier
epoch only on exact ties. Validation one-step error is recorded but does not
select the checkpoint. Test/OOD evaluation requires the completed, hashed
selection. Training does not read held-out evaluation results.

## Interpretation frozen before training

The three requested cases are recorded in the run protocol before any gate or
full training. Using held-out mean final ID autoregressive relative L2:

1. D substantially beats B/C and approaches A: supports an accuracy tradeoff
   introduced by the Stage 5 rollout objective/training setup under this protocol.
2. D is similarly poor to B/C: supports the shared lower-learning-rate/budget
   protocol as the more plausible dominant explanation.
3. D lies between A and B/C: mixed evidence; both budget and objective may matter.

To avoid choosing the meaning of these terms after seeing D, apply this fixed
descriptive ordering: case 1 if `D <= 0.8*min(B,C)` and `D <= 1.2*A`; otherwise
case 2 if `abs(D/mean(B,C)-1) <= 0.1` and `D > 1.2*A`; otherwise case 3 if
`A < D < min(B,C)`. Outcomes outside these regions are reported as such.
These are descriptive thresholds, not significance tests or proof of causality.
No threshold will change after evaluation. The one-seed, small-split study
cannot separate every remaining optimization confound.

## Sanity gate and provenance

The gate uses the first transition from each of the first eight training
trajectories `[0,50,21,20,38,10,45,59]`. With the same Adam settings, it has at
most 300 updates and is checked every 25. Passing requires finite gradients,
at least 95% normalized-MSE reduction, and mean physical relative L2 at most 5%.
A failed gate stops the program before full training.

Initialization is required to match Stage 5's parameter-state SHA-256 exactly:
`6569c2e401830b7eb13ddd91e1f118005b4863f67a188952f7b5e83fde8a4884`.
The historical checkpoint, split, normalization, training archive, and source
hashes are checked. The initial protection snapshot covers 207 existing
files/artifacts, including the prior 98 protected Stage 1–4 files and all
Stage 5 source, public results, and local runs. README/package discovery are
the only existing project files permitted to change.

## Optional E policy

After D completes, estimate E's 100-epoch runtime from D's measured mean epoch
time. Two additional CPU minutes is the predeclared ceiling for a clearly
modest extra experiment. If the estimate exceeds it, omit E and report the
estimate. D is the required study; E is not necessary for completion.

## Commands and output organization

From the repository root, with the frozen Stage 3–5 local artifacts available:

```bash
.venv/bin/python -m pip install -e '.[test,evaluation]'
.venv/bin/python -m examples.train_stage6
.venv/bin/python -m examples.evaluate_stage6
.venv/bin/python -m examples.report_stage6
.venv/bin/python -m pytest -q
```

Training/evaluation refuse to overwrite existing run directories. Reporting
requires new public JSON/CSV paths. Once public summaries exist, reproduce them
and regenerate the ignored figures with a fresh export directory:

```bash
.venv/bin/python -m examples.report_stage6 --public runs/stage6/report_copy/stage6_summary.json
```

For a separate full reproduction, use `--output runs/stage6_reproduction` for
training and `--run runs/stage6_reproduction` for evaluation and reporting,
with a new `--public` destination. No old artifacts need to be deleted.

The new `stage6_control/` package contains fixed configuration/interpretation,
provenance/pair reuse, one-step training/gate/validation, and reporting modules.
The three `examples/*stage6.py` entry points separate training, frozen
evaluation, and reporting. Tests live in `tests/test_stage6_*.py`.
Only README and package discovery in `pyproject.toml` change outside the new
Stage 6 files. Checkpoints, full histories, protocol/verification records,
predictions, and figures stay under ignored `runs/stage6/`.

The budget plot uses saved per-epoch validation final AR errors, never test
errors. Its wall-time axis sums training plus validation seconds; Stage 5 did
not record optimization-only seconds separately. Gates, final evaluation,
and most checkpoint I/O are excluded from that axis. D also separately records
training-phase time, including data loading, and end-to-end full-run runtime.
No historical run is edited.

Supervised-field counts in the central table refer to complete training runs,
not just their selected checkpoints, and exclude discarded gate training.
Selected-epoch counts are reported separately. The three lambda candidates
needed to select C are also not included in C's per-model training budget;
Stage 5's total selection cost was higher than one candidate's cost.

## Observed sanity gate

All pre-training checks passed, including exact initialization hash, frozen
split/normalization reuse, pair shapes, and finite gradients. The eight-pair
gate passed at update 150 in 18.39 seconds:

| Metric | Initial | Final |
| --- | ---: | ---: |
| Normalized MSE | 1.3157627583 | 0.0036990317 |
| Physical mean relative L2 | 88.3053% | 4.6422% |

The MSE decreased by about 99.72%. Full training started from a newly created
model with the exact same initial parameter hash, not from these gate weights.

## Training completion and checkpoint selection

D completed all **60 epochs, 3,600 optimizer updates, and 28,800 supervised
field comparisons**. Training plus validation/checkpoint bookkeeping took
507.07 seconds (8.45 minutes); training-phase time including data loading totaled
453.20 seconds. The discarded gate took another 18.39 seconds.
The reference device is Apple M1 CPU, four intra-op threads, Python 3.10.8,
NumPy 2.2.6, and PyTorch 2.14.0. There is no GPU benchmarking.

Validation selected **epoch 50** with final AR L2 **12.5971%**, mean one-step
L2 **1.7685%**, and final target mass error **4.7459%**. This checkpoint had
processed 24,000 supervised fields through 3,000 updates. Neither test nor OOD
data contributed to selection. The checkpoint is `runs/stage6/best.pt`, SHA-256:

```text
cf8899453cadf2443961fd65daa736269ed671dd26a375fd3edc1159ae56ca33
```

Late optimization fluctuated. Epoch 60 had training MSE 0.000627131 and
validation final AR error 21.2326%, substantially worse than the selected
epoch. All values remained finite. The existing best-checkpoint policy
preserved epoch 50; no learning-rate or scheduler intervention was made.
The full, unsmoothed history is retained, including this final-epoch failure.

Optional E is estimated at **845.11 seconds (14.09 minutes)**,
using 100 times D's mean full-run epoch time, before any additional gate or
evaluation. This exceeds the 120-second ceiling and would materially increase
runtime. **E was not run.** The estimate is approximate, not a timing guarantee.

## Budget-comparison diagnostics

These are raw per-epoch **validation** final AR errors, not held-out test
errors and not smoothed/best-so-far curves. Exact common-budget points are
retained in the public JSON; no interpolation is required for these rows.

| Matched axis | Value | B epoch | C epoch | D epoch | B AR | C AR | D AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch | 20 | 20 | 20 | 20 | 21.670% | 21.565% | 15.883% |
| Optimizer updates | 720 | 20 | 20 | 12 | 21.670% | 21.565% | 18.105% |
| Supervised fields | 14,400 | 10 | 10 | 30 | 23.349% | 23.047% | 14.256% |
| Supervised fields | 23,040 | 16 | 16 | 48 | 21.197% | 20.488% | 12.993% |
| Supervised fields | 28,800 | 20 | 20 | 60 | 21.670% | 21.565% | 21.233% |

D already beats B/C's full-run raw endpoints at the same 720-update count,
having processed only 5,760 training fields. It also has lower validation
error at the intermediate matched-field points. These observations support
a training-setup difference beyond simply providing more updates, although
the different examples, horizon weighting, and gradient paths remain.

The final matched-field endpoint alone looks similar because of D's epoch-60
spike. Reporting only D's best epoch would conceal that optimization behavior;
reporting only its last epoch would disregard the predeclared selection rule.
Both views are therefore retained. Best validation checkpoints were reached
at cumulative epoch wall times 373.22s (B), 366.00s (C), and 421.09s (D).
Their best errors were 21.197%, 20.488%, and 12.597%, respectively.
Complete epoch-time sums were 468.47s, 454.68s, and 506.32s. D's full run was
somewhat slower, despite equal supervised-field exposure. None of these
measurements establishes exact FLOP equivalence.

`runs/stage6/figures/budget_comparison.png` presents all B/C/D validation
histories against epoch, updates, supervised fields, and measured wall time.
The one-step loss/validation histories for D, exact matched-budget values,
and hashes of the unchanged B/C histories are retained in the summary.

## Frozen held-out comparison

All four models were evaluated anew with the unchanged Stage 4 evaluator on
the existing, hash-verified archives. A/B/C autoregressive diagnostics and
teacher-forced L2 curves reproduce Stage 5 within the fixed verification
tolerances (rtol 1e-6, atol 1e-9). No model or dataset was modified for evaluation.
Errors below are physical relative errors, averaged over trajectories. The
one-step measure averages ten teacher-forced transitions; final AR means t=1.

| Model | Objective | Supervised fields, full run | ID one-step | ID final AR | ID mass error | High-velocity AR | 64 AR | 128 AR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Historical one-step | 48,000 | 2.358% | 17.134% | 8.188% | 40.581% | 14.837% | 14.947% |
| B | K=5 rollout | 28,800 | 6.352% | 28.848% | 25.676% | 55.181% | 24.287% | 24.347% |
| C | K=5 rollout + mass | 28,800 | 6.381% | 28.215% | 24.705% | 53.687% | 24.290% | 24.355% |
| D | Matched one-step | 28,800 | 2.374% | 17.409% | 7.041% | 38.716% | 14.160% | 14.265% |

| Model | Selected epoch | Fields processed by selected checkpoint |
| --- | ---: | ---: |
| A | 31 | 14,880 |
| B | 16 | 23,040 |
| C | 16 | 23,040 |
| D | 50 | 24,000 |

The public [JSON](results/stage6_summary.json) preserves exact splits,
normalization, provenance, every horizon, all distributions, physical
diagnostics, and budget curves. The [CSV](results/stage6_comparison.csv)
contains the central comparison and selected-checkpoint field counts.

### ID rollout and feedback

| Time | A AR L2 | B AR L2 | C AR L2 | D AR L2 |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 2.520% | 6.214% | 6.215% | 2.527% |
| 0.2 | 4.557% | 9.761% | 9.663% | 4.635% |
| 0.3 | 6.431% | 12.850% | 12.699% | 6.562% |
| 0.4 | 8.197% | 15.642% | 15.446% | 8.366% |
| 0.5 | 9.874% | 18.218% | 17.968% | 10.072% |
| 0.6 | 11.470% | 20.617% | 20.301% | 11.690% |
| 0.7 | 12.990% | 22.863% | 22.473% | 13.227% |
| 0.8 | 14.438% | 24.977% | 24.505% | 14.690% |
| 0.9 | 15.818% | 26.969% | 26.415% | 16.083% |
| 1.0 | 17.134% | 28.848% | 28.215% | 17.409% |

| Final-horizon metric | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forced L2 | 2.248% | 6.678% | 6.739% | 2.264% |
| Feedback ratio | 7.622x | 4.320x | 4.187x | 7.691x |

Feedback divides mean AR error by mean TF error at the same horizon, guarded
near zero; it is not a success criterion. D has a ratio close to A. The lower
B/C ratios are accompanied by much larger teacher-forced and rollout errors.
Final ID persistence error remains 76.550%; every model beats that aggregate.

### Final ID physical diagnostics

| Diagnostic | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Target mass error | 8.188% | 25.676% | 24.705% | 7.041% |
| Signed mass drift | -1.836% | -13.435% | -13.542% | -0.670% |
| Accumulated absolute mass changes | 9.108% | 29.143% | 27.678% | 8.036% |
| Negative-cell fraction | 5.914% | 23.813% | 27.161% | 7.495% |
| Negative cells below tolerance | 5.911% | 23.813% | 27.161% | 7.495% |
| Negative-mass fraction | 0.069% | 1.084% | 1.448% | 0.123% |
| Mean trajectory minimum | -0.002349 | -0.012459 | -0.014062 | -0.004184 |
| Mean trajectory maximum | 1.046047 | 0.967353 | 0.960910 | 1.042953 |

Mass drift is relative to initial net mass; accumulated changes sum absolute
step-to-step mass increments before dividing by initial absolute net mass.
Negative mass is relative to the initial absolute-field integral. The negative
cell tolerance remains `1e-7 * max(1,max(abs(u0)))`. No correction is applied.
D improves ID integral conservation over A but has more negative cells and
negative mass. Signed drift alone can hide cancellation across trajectories.

### Fresh-ID and parameter-OOD

Final AR relative L2 on the exact frozen Stage 4 distributions; velocity
quadrants label the signs of `(cx,cy)`:

| Distribution | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| fresh_id | 15.412% | 23.953% | 24.276% | 14.398% |
| high_nu | 14.059% | 24.643% | 24.458% | 14.034% |
| high_velocity | 40.581% | 55.181% | 53.687% | 38.716% |
| velocity_pp | 36.782% | 63.733% | 62.077% | 35.715% |
| velocity_pn | 50.110% | 50.993% | 51.050% | 47.947% |
| velocity_np | 34.146% | 40.669% | 39.561% | 34.006% |
| velocity_nn | 41.286% | 65.329% | 62.059% | 37.194% |

Final target-mass errors for these distributions:

| Distribution | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| fresh_id | 8.750% | 19.865% | 18.592% | 7.462% |
| high_nu | 6.103% | 18.435% | 18.238% | 7.810% |
| high_velocity | 13.196% | 39.191% | 37.990% | 11.851% |
| velocity_pp | 11.181% | 58.970% | 55.413% | 9.854% |
| velocity_pn | 17.882% | 23.489% | 25.488% | 20.041% |
| velocity_np | 9.199% | 14.164% | 15.497% | 8.813% |
| velocity_nn | 14.524% | 60.140% | 55.562% | 8.697% |

D improves high-velocity field error relative to A on this sample, but a
38.716% combined error remains substantial. Its positive-negative quadrant
has 47.947% field error and 20.041% mass error. High-nu field error is close
between A and D, while D has worse high-nu mass error. The control does not
establish uniformly better conservation or solve parameter extrapolation.

### Matched 64/128 resolution transfer

Both grids use identical frozen weights and normalization, with matched
physical initial conditions and coefficients. Native-grid errors appear in
the central table. No retraining or resolution-specific adjustment occurs.

| Model | Native 128 minus 64 error, percentage points | Shared-grid AR prediction relative L2 |
| --- | ---: | ---: |
| A | 0.1096 | 1.138e-07 |
| B | 0.0601 | 1.282e-07 |
| C | 0.0646 | 1.294e-07 |
| D | 0.1057 | 1.104e-07 |

Shared-grid differences compare the 64 prediction with every second point
of the 128 prediction, normalized by the latter. Close model predictions
across grids do not imply continuum accuracy: the 128 reference remains a
discretized solver solution. Native-error changes also reflect differences
between the numerical references.

## Predeclared interpretation and limitations

**Case 1 applies.** D's final ID error is 39.65% lower than B's and
38.30% lower than C's, while only 0.275 percentage points above A
(1.61% higher in relative terms). Both frozen case-1 thresholds hold.
This supports an accuracy tradeoff from the Stage 5 rollout objective/training
setup under the tested protocol. Lower learning rate and fewer supervised
fields alone do not explain B/C's degradation: a one-step model with those
settings and the same selection criterion approaches historical A.

This is not proof that rollout loss itself is intrinsically inferior. D has
more updates, different horizon multiplicities, ground-truth inputs at every
pair, and a different gradient graph. The matched-update validation comparison
adds evidence but does not remove all these differences. There is only one
seed and eight original ID test trajectories, with no confidence interval or
statistical-significance claim. B/C additionally underwent a validation-only
lambda selection for C. E was omitted, so historical selection-versus-schedule
effects are not separately isolated. Stage 5 remains an unchanged negative
result; Stage 6 adds context rather than rewriting it.

No nonfinite training, validation, or held-out rollout failures occurred.
D's final-epoch optimization spike and its worse ID negativity than A remain
material caveats. The only plotting notices were local Matplotlib font-cache
creation and the macOS static-font-registry message.

Evaluation including prediction saves took 49.68s (A), 50.09s (B), 49.45s (C), 49.83s (D).
The ignored `budget_comparison.png` and `id_comparison.png` figures were
visually inspected for labels, scales, and clipping. The latter shows both
ID rollout error and target mass error across all four models.

## Regression and protected-file verification

The complete Stage 1–6 suite reports **163 passed in 2.76 seconds**: all 150
existing tests plus 13 new tests. New coverage checks frozen split and
normalization reuse, pair membership/shapes, initialization reproducibility,
field/update budgets, validation-only checkpoint ordering, unchanged Stage 5
rollout compatibility, protected-file mutation detection, fixed interpretation
cases, D checkpoint provenance, failed-gate blocking, and read-only use of
historical budget histories.

All **207 protected files/artifacts** match the pre-implementation SHA-256
snapshot, including all 98 previously protected Stage 1–4 files. Stage 5 source,
public documentation/results, checkpoints, histories, plots, evaluation outputs,
and verification records are unchanged. D's initialization matches Stage 5;
frozen training-source hashes, final Stage 6 source hashes, and historical
budget-history hashes all verify. The local verification record is
`runs/stage6/protected_verification.json`.

Created:

- `stage6_control/{__init__,config,provenance,training,reporting}.py`
- `examples/{train,evaluate,report}_stage6.py`
- `tests/test_stage6_control.py` and `tests/test_stage6_reporting.py`
- `docs/stage6_matched_control.md`
- `docs/results/stage6_summary.json` and `docs/results/stage6_comparison.csv`

Modified: README (status/link/commands) and `pyproject.toml` (package discovery).
All generated run outputs remain ignored, as do existing datasets and personal
learning notes. Nothing was staged, committed, or pushed; this implementation
awaits review.
