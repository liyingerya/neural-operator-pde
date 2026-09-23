"""Validated configuration for small periodic Gaussian trajectory datasets."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling configuration; times include zero and are equally spaced.

    Bounds describe independent continuous uniform distributions. Blob count
    is discrete uniform from one through three. No ML splits are generated.
    """

    num_trajectories: int = 64
    grid_size: int = 64
    num_snapshots: int = 11
    t_final: float = 1.0
    seed: int = 2026
    cx_range: tuple[float, float] = (-1.0, 1.0)
    cy_range: tuple[float, float] = (-1.0, 1.0)
    nu_range: tuple[float, float] = (0.005, 0.05)
    width_range: tuple[float, float] = (0.4, 0.8)
    amplitude_range: tuple[float, float] = (0.5, 1.5)
    safety: float = 0.9

    def __post_init__(self) -> None:
        for name, minimum in (("num_trajectories", 1), ("grid_size", 3),
                              ("num_snapshots", 2), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.t_final) or self.t_final <= 0:
            raise ValueError("t_final must be finite and positive")
        if not np.isfinite(self.safety) or not 0 < self.safety <= 1:
            raise ValueError("safety must be in (0, 1]")
        for name in ("cx_range", "cy_range", "nu_range", "width_range", "amplitude_range"):
            bounds = np.asarray(getattr(self, name))
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must contain two increasing finite bounds")
        if self.nu_range[0] < 0 or self.width_range[0] <= 0 or self.amplitude_range[0] <= 0:
            raise ValueError("diffusivity must be nonnegative; width and amplitude positive")

    def snapshot_times(self) -> np.ndarray:
        """Return strictly increasing physical times, including zero and T."""
        return np.linspace(0.0, self.t_final, self.num_snapshots, dtype=np.float64)
