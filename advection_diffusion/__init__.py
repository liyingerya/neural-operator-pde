"""NumPy solver for periodic two-dimensional advection–diffusion."""

from .initial_conditions import periodic_gaussian
from .solver import periodic_grid, rk4_step, solve, spatial_operator, stable_timestep

__all__ = [
    "periodic_gaussian", "periodic_grid", "rk4_step", "solve",
    "spatial_operator", "stable_timestep",
]
