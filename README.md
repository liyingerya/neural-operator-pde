# Neural Operator for 2D Advection–Diffusion

A Scientific ML study of when a Fourier Neural Operator succeeds—and fails—on
a parameterized 2D advection–diffusion PDE. I built and verified the numerical
solver, generated reproducible trajectories, trained a conditioned FNO from
scratch in PyTorch, and evaluated rollout behavior, parameter extrapolation,
physical conservation, grid transfer, and runtime.

[Technical report](docs/final_project_report.md) · [Results](docs/results/final_model_comparison.csv) · [Reproduction guide](docs/reproducibility.md) · [Provenance](docs/stage7_provenance.md)

## Why this project

Can a learned operator predict a parameterized periodic advection–diffusion PDE
reliably beyond one-step interpolation? A verified solver provides reproducible
labels; trajectory-level splits prevent snapshot leakage; frozen evaluations
and a matched one-step control make the limitations visible.

## Key results

Errors are physical relative L2 unless labeled otherwise. A is the historical
one-step baseline; D is the matched one-step control. Final rollout means ten
steps to `t=1`.

| Measurement | Frozen result |
| --- | --- |
| A: mean one-step ID error | **2.358%**, versus **8.439%** one-step persistence |
| A: final ID rollout error | **17.134%**; final feedback ratio **7.62×** |
| A: high-velocity OOD rollout error | **40.581%** |
| D: final ID rollout error | **17.409%**, versus **28.848% / 28.215%** for rollout / rollout+mass models |
| A: matched 64 → 128 native-grid error | **14.837% → 14.947%**, with no retraining |
| Numerical verification | Second-order spatial and fourth-order temporal convergence checked against Fourier references |
| CPU interval latency, batch 1 | Solver **0.345 ms**, FNO end-to-end **8.129 ms** on Apple M1: FNO **23.57× slower** |
| A: final ID mass error | **8.188%**; conservation is not guaranteed |

These are measured findings on a small workload, including negative results—not
claims of a faster or generally reliable PDE replacement.

![Historical baseline rollout error](docs/figures/rollout_error.png)

## Pipeline

Periodic PDE → verified finite-difference/RK4 solver → reproducible trajectories
→ conditioned FNO → one-step evaluation → autoregressive/OOD/physical diagnostics
→ controlled training ablations and matched control.

## Model

FNO is a natural baseline here because the PDE is periodic and its dynamics
are naturally represented across spatial Fourier modes.

The network maps `[u(t), cx, cy, nu]` to `u(t+0.1)`. Three scalar coefficients
are broadcast over the periodic grid; no coordinate channels are added.
Four blocks combine Fourier spectral convolution and local pointwise maps,
with 12 retained modes, width 32, and GELU activations. The model has
1,186,209 trainable tensor elements; complex weights correspond to 2,365,857
real scalar components.

## Scientific findings

- Accurate single steps still accumulate substantial feedback error.
- High-velocity extrapolation is harder here than high-diffusivity extrapolation.
- Field accuracy, mass conservation, and positivity are distinct properties.
- Rollout and soft-mass training underperformed the historical baseline.
- A matched one-step control recovered near-baseline accuracy; unequal update
  counts and gradient paths remain confounds, not proof of universal inferiority
  of rollout training.
- Similar 64/128 errors demonstrate a specific grid-transfer result, not continuum accuracy.
- The FNO is slower than this inexpensive CPU solver; acceleration is workload-dependent.

![Historical, ablation, and matched-control comparison](docs/figures/stage6_ablation.png)

## Reproduce

Python 3.10+; run from the repository root. NumPy supports the solver/data code;
PyTorch and Matplotlib are optional ML/evaluation dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,evaluation]'
python -m pytest -q
```

| Task | Entry point |
| --- | --- |
| Stage 1 demo | `python -m examples.gaussian_demo --output /tmp/gaussian_demo.npz` |
| Stage 2 development data | `python -m examples.generate_dev_dataset` |
| Stage 3 baseline training | `python -m examples.train_fno` |
| Stage 4 frozen evaluation | `python -m examples.evaluate_stage4 --phase id` |
| Stage 5 ablation | `python -m examples.train_stage5` |
| Stage 6 matched control | `python -m examples.train_stage6` |
| Final public figures/table, no training | `python -m examples.build_final_report` |

The [reproduction guide](docs/reproducibility.md) gives the complete ordered
commands and prerequisites. Datasets, checkpoints, run histories, and experimental
plots are intentionally omitted from Git. Exact Stage 4–6 replay requires the
approved local artifacts: fresh training is not guaranteed to reproduce their
hashes. Tests and final figure generation work from public files alone.

## Repository structure

- `advection_diffusion/`, `dataset_generation/`: numerical reference and trajectories.
- `neural_operator/`: PyTorch data, normalization, FNO, training, and metrics.
- `stage4_evaluation/`, `stage5_training/`, `stage6_control/`: frozen experiments.
- `examples/`, `tests/`: executable workflows and regression checks.
- `docs/`: [final report](docs/final_project_report.md), historical reports, compact results, and five curated figures.

## Limitations

Training uses smooth Gaussian mixtures, only 48 training trajectories, and one
primary seed. The eight-trajectory ID test set supports no broad confidence
claim. Labels are discretized solver solutions; neither conservation nor
positivity is enforced. CPU timing is specific to the implementation, hardware,
and workload. There is no claim of universal OOD generalization or resolution
invariance. The scientific stages are complete; final presentation preserves
their frozen results.
