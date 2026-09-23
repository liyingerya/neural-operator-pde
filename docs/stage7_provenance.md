# Final synthesis provenance and release audit

Stage 7 is presentation and reproducibility work, not a new scientific experiment.
The working tree was clean before changes at
`cfb580e134f8e30de3c6659b3ec29cf234c07cb2` (Stage 6). No model was trained,
retrained, tuned, or newly evaluated for the synthesis; optional E was not run.
Historical source, reports, numerical summaries, datasets, and checkpoints stay
unchanged. The inexpensive demo/data smoke checks described below are command
validation, not new evidence for the scientific conclusions.

## Scientific source revisions

| Frozen work | Source revision |
| --- | --- |
| Verified solver | `ffaf8c951d5a32a84f8e28c73cca0a7f68db9da6` |
| Trajectory generation | `c99393d50ae220c71385ad61ba8a7489b2f0c37f` |
| Historical FNO baseline | `9f1cc21d5187c50c4658f5cb36cae0b14dd5f1b1` |
| Frozen rollout/OOD evaluation | `925d58aa2b8448f4e12aa80e7131d1294b8d8938` |
| Rollout/mass ablations | `fb31c8c0744370059510616ecbf4090923111558` |
| Matched one-step control | `cfb580e134f8e30de3c6659b3ec29cf234c07cb2` |

Run manifests may name the earlier repository HEAD that existed while a stage
was being implemented. Their source-file hashes identify the actual code bytes;
a manifest's Git revision alone is not a claim that all run code was committed
at training time. This historical detail is preserved rather than rewritten.

## Frozen values and consistency

Before presentation changes, the existing checkpoint loaders verified approved A,
selected B/C, and selected D. The previous protected-file snapshot, Stage 4
artifact hashes, Stage 5/6 source hashes, and saved budget-history hashes passed.
An expanded read-only snapshot records **313 protected files/artifacts**, including
all previously protected sources/results and the completed Stage 6 run. README is
intentionally excluded because this stage rewrites it; historical reports are not.

Across public Stage 4/5/6 summaries, **1,630 horizon/diagnostic arrays** match the
saved local evaluation outputs, checked without inference. Stage 5/6 primary CSV
metrics agree with public JSON. The rounded Stage 3 FNO and one-step persistence
results agree with its saved evaluation. The new final table copies Stage 6's
consistent A/B/C/D re-evaluation values; it does not change or average across
historical results to improve a metric.

The [final summary](results/final_summary.json) records SHA-256 values for its
tracked inputs. Sources are:

- [Stage 4 public summary](results/stage4_summary.json): baseline horizon curves,
  rollout persistence, CPU timing, and hardware context.
- [Stage 6 public summary](results/stage6_summary.json): all A/B/C/D final model
  comparisons and training/selected-checkpoint field counts.
- [Stage 3 report](stage3_fno.md): published rounded one-step persistence result.
- [Frozen solver tests](../tests/test_solver.py): numerical-verification diagram.

One-step persistence uses each true preceding field; rollout persistence retains
`u0`. They have distinct definitions and are kept separate. Stage 3's float32
training-target conversion and later float64-reference metrics are also identified
in the final report. Percentage conversion, rounding for display, and copying
existing aggregate fields are the only numerical transformations in the builder.

## Curated public figures

Exactly five PNGs are included under `docs/figures/`:

| Figure | Source and selection |
| --- | --- |
| [Numerical verification](figures/numerical_verification.png) | Map of frozen analytic, convergence, and conservation checks; no invented error curve |
| [Rollout error](figures/rollout_error.png) | All original ID horizons for A: teacher forcing, autoregression, and rollout persistence |
| [OOD comparison](figures/ood_comparison.png) | All four requested regime aggregates for A and D; B/C remain visible in the ablation table/figure |
| [Ablation comparison](figures/stage6_ablation.png) | A/B/C/D ID one-step, final rollout, and mass error, including the negative results |
| [Resolution transfer](figures/resolution_transfer.png) | All four models, both matched grids, native-grid final error |

No attractive individual trajectory was selected for the final figure set. The
figures are rebuilt by [the tracked synthesis script](../examples/build_final_report.py)
from public files alone. Historical experimental figures remain ignored. The
small curated PNGs are intentional public artifacts, not model/data binaries.

## Audit scope

The audit covers current tracked files and intended Stage 7 additions, their
sizes, prohibited data/checkpoint/cache artifacts, personal notes, local absolute
paths/usernames, and relative documentation links. Historical reports remain
available as dated stage records: statements describing a stage's then-current
scope are not silently revised into new claims. The current README and final
report are the primary entry points.

Personal notes remain local-only through `.git/info/exclude`, including the new
portfolio/interview material. No personal-note rule was added to tracked
`.gitignore`. Local detailed audit logs and the full protected hash snapshot are
under ignored `runs/stage7/`. No commit or push is performed during finalization.

## Completed release audit

- All 313 protected files retain their pre-finalization SHA-256 hashes.
- The full suite passes in the working checkout: **167 passed in 2.97 s**.
  A clean Stage 6 clone overlaid with the intended Stage 7 public files also
  passes: **167 passed in 2.95 s**.
- All 19 clean-checkout checks pass: package installation, installed-package
  imports, the full suite, the Stage 1 demo, Stage 2 development-data generation,
  11 entry-point help checks, and the public-only synthesis rebuild. Installation
  used the existing dependency environment; fresh network dependency resolution
  was not tested. No training or model evaluation was run.
- Regenerated development scientific arrays match the frozen archive exactly.
  Rebuilt final JSON and CSV outputs match the curated outputs byte-for-byte.
- All 45 relative public documentation links resolve. The five final figures
  were visually inspected and total 289,936 bytes (about 283 KiB).
- The prospective public tree is about 1.64 MiB. Its largest file is the existing
  Stage 6 JSON summary (432,357 bytes); no public file exceeds 1 MiB. Generated
  datasets, checkpoints, histories, experimental plots, verification records,
  caches, and personal notes are excluded from the public file set.
- No local home paths/usernames or unfinished placeholder markers were found
  in the prospective public text files.
- README is the only modified previously tracked file. Twelve new public files
  supply the final report, reproduction guide, provenance note, two result
  tables, five figures, synthesis script, and lightweight publication tests.
  The portfolio note and its local exclude entry remain local-only.

The scientific checkout revision remains the recorded Stage 6 HEAD. These
checks prepare the changes for review; they do not create a release commit.
