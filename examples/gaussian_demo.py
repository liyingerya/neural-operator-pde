"""Run from the repository root: python -m examples.gaussian_demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from advection_diffusion import periodic_gaussian, periodic_grid, solve, stable_timestep


def main() -> None:
    """Evolve one Gaussian, print diagnostics, and save initial/final fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("gaussian_demo.npz"))
    args = parser.parse_args()
    n, cx, cy, nu, t_final = 64, 1.0, -0.5, 0.01, 1.0
    x, y = periodic_grid(n)
    initial = periodic_gaussian(x, y)
    dt = stable_timestep(n, cx, cy, nu)
    final = solve(initial, t_final, cx, cy, nu, dt)
    # Exact continuous solution: translated center and diffusive broadening.
    sigma = 0.4
    width = np.sqrt(sigma**2 + 2.0 * nu * t_final)
    exact = periodic_gaussian(
        x, y, center=(np.pi + cx * t_final, np.pi + cy * t_final),
        sigma=width, amplitude=sigma**2 / width**2,
    )
    cell_area = (2.0 * np.pi / n)**2
    initial_mass, final_mass = initial.sum() * cell_area, final.sum() * cell_area
    print(f"Grid: {n} x {n}; t_final={t_final:g}; dt={dt:.8g}")
    print(f"Initial mass: {initial_mass:.16g}")
    print(f"Final mass:   {final_mass:.16g}")
    print(f"Mass change: {final_mass - initial_mass:.3e}")
    print(f"Final range: [{final.min():.6e}, {final.max():.6e}]")
    print(f"Relative L2 error vs analytic Gaussian: {np.linalg.norm(final-exact) / np.linalg.norm(exact):.6e}")
    np.savez(args.output, x=x, y=y, initial=initial, final=final, exact=exact,
             cx=cx, cy=cy, nu=nu, t_final=t_final, dt=dt)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
