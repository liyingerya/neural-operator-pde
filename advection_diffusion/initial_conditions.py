"""Smooth Gaussian initial data on a periodic square."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def periodic_gaussian(
    x: ArrayLike, y: ArrayLike,
    center: tuple[float, float] = (np.pi, np.pi),
    sigma: float = 0.4, amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate a periodized isotropic Gaussian of width sigma.

    Sum translated Gaussian copies in each direction, retaining images
    beyond nine standard deviations. Omitted tails are below float64
    precision relative to the peak. amplitude scales the image sum; the
    peak need not equal amplitude when broad images overlap. Coordinates
    can be broadcast together and may lie outside [0, 2*pi).
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if np.shape(center) != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain two finite coordinates")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinates must be finite")
    period = 2.0 * np.pi
    images = int(np.ceil(9.0 * sigma / period)) + 1

    def image_sum(coordinate: NDArray[np.float64], location: float) -> NDArray[np.float64]:
        displacement = (coordinate - location + np.pi) % period - np.pi
        result = np.zeros_like(displacement)
        for shift in range(-images, images + 1):
            result += np.exp(-0.5 * ((displacement + shift * period) / sigma)**2)
        return result

    return amplitude * image_sum(x, center[0]) * image_sum(y, center[1])
