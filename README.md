# Neural Operator for 2D Advection–Diffusion

Stage 1: a verified NumPy solver for the periodic equation
`du/dt + cx du/dx + cy du/dy = nu (d²u/dx² + d²u/dy²)`.

## Run

Python 3.10+ is required. NumPy is the only runtime dependency; pytest is
used for verification. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
python -m examples.gaussian_demo --output /tmp/gaussian_demo.npz
```

The demo reports mass, extrema, and error against the analytic Gaussian,
then saves coordinates, initial/final fields, analytic reference, and
parameters in a single NumPy archive. It requires no plotting library.

## Numerical method

The domain is `[0, 2π) × [0, 2π)`, with grid spacing `h = 2π/n` and no
duplicate endpoints. Arrays have shape `(n, n)`: axis 0 is x, axis 1 is y.
Constant velocities and nonnegative diffusivity are supported.

Centered second-order finite differences use `np.roll` for periodic neighbors:

```text
Dx(u)[i,j]  = (u[i+1,j] - u[i-1,j]) / (2h)
Dxx(u)[i,j] = (u[i+1,j] - 2u[i,j] + u[i-1,j]) / h²
L(u)       = -cx Dx(u) - cy Dy(u) + nu (Dxx(u) + Dyy(u))
```

Classical explicit RK4 evaluates L at four stages per timestep. Accuracy is
second order in space and fourth order in time for smooth solutions. The
initial Gaussian is summed over periodic images; tails are truncated beyond
nine standard deviations, below double-precision significance at the peak.

For a discrete Fourier mode, `lambda = -a - i b`, with

```text
a = (4 nu / h²) [sin²(theta_x/2) + sin²(theta_y/2)] <= 8 nu / h²
|b| = |(cx sin(theta_x) + cy sin(theta_y)) / h| <= (|cx| + |cy|) / h.
```

RK4 has stability polynomial `R(z) = 1 + z + z²/2 + z³/6 + z⁴/24`.
The rectangle `-1 <= Re(z) <= 0`, `|Im(z)| <= 1` lies in its stability
region. The timestep helper therefore uses the sufficient bound

```text
dt = safety * min(h / (|cx| + |cy|), h² / (8 nu))
```

The default safety factor is 0.9. Zero denominators give infinite bounds.
Pure advection is supported; zero coefficients leave the field unchanged.
Supplied timesteps must satisfy the bound with safety=1; this conservative
check can reject steps that pass a sharper spectral analysis. The final
step is shortened to reach the requested final time.

This is discrete L2 stability, not positivity or monotonicity preservation.
Small oscillations can occur, especially for poorly resolved pulses.
Pure-advection grid-scale checkerboard modes have zero centered derivative.
Periodic differences conserve total mass up to floating-point roundoff.

## API and files

```python
from advection_diffusion import periodic_grid, periodic_gaussian, solve

x, y = periodic_grid(64)
u0 = periodic_gaussian(x, y, sigma=0.4)
u = solve(u0, t_final=1.0, cx=1.0, cy=-0.5, nu=0.01)
```

- `advection_diffusion/solver.py`: grid, spatial operator, timestep helper,
  single RK4 step, and final-state integration. Inputs are not mutated.
- `advection_diffusion/initial_conditions.py`: smooth periodic Gaussian.
- `advection_diffusion/__init__.py`: public API exports.
- `examples/gaussian_demo.py`: diagnostics and NumPy output.
- `tests/`: constants, mass conservation, Fourier analytic comparisons,
  second-order spatial and fourth-order temporal convergence, Gaussian
  periodicity, discrete-mode stability, and invalid input checks.
- `pyproject.toml`: package metadata, dependencies, and pytest configuration.

## Status

Stages 1–3 are implemented: the verified solver, reproducible trajectory
datasets, and a conditioned one-step PyTorch Fourier Neural Operator baseline.

- [Stage 2 dataset schema and generation](docs/stage2_dataset.md)
- [Stage 3 training protocol, verification, and results](docs/stage3_fno.md)

Stage 3 uses an optional ML dependency: `pip install -e '.[test,ml]'`.
Run it with `python -m examples.train_fno`; the eight-pair overfit gate must
pass before full training starts. Stage 4 rollout, OOD, resolution-transfer,
and benchmarking work is not implemented.
