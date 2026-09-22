"""Centered finite differences and explicit RK4 on [0, 2*pi)^2.

Fields have shape (n, n); axis 0 is x and axis 1 is y. Stability is
in the discrete L2 sense, not a guarantee of positivity or monotonicity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Field = NDArray[np.float64]


def _spacing(n: int) -> float:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 3:
        raise ValueError("n must be an integer >= 3")
    return 2.0 * np.pi / n


def _parameters(cx: float, cy: float, nu: float) -> None:
    if not np.all(np.isfinite([cx, cy, nu])) or nu < 0:
        raise ValueError("cx and cy must be finite; nu must be finite and nonnegative")


def _field(u: ArrayLike) -> Field:
    if np.iscomplexobj(u):
        raise ValueError("u must be real-valued")
    field = np.asarray(u, dtype=float)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("u must be a square two-dimensional field")
    _spacing(field.shape[0])
    if not np.all(np.isfinite(field)):
        raise ValueError("u must contain only finite values")
    return field


def periodic_grid(n: int) -> tuple[Field, Field]:
    """Return x, y coordinate arrays of shape (n, n), excluding 2*pi."""
    coordinates = np.arange(n) * _spacing(n)
    return np.meshgrid(coordinates, coordinates, indexing="ij")


def stable_timestep(
    n: int, cx: float, cy: float, nu: float, safety: float = 0.9,
) -> float:
    """Return safety * min(h/(|cx|+|cy|), h**2/(8*nu)).

    This sufficient RK4 bound places all dt*lambda in the stable rectangle
    [-1, 0] + i[-1, 1]. Zero denominators give infinite bounds; a stationary
    equation returns infinity. Pure advection (nu=0) is supported.
    """
    h = _spacing(n)
    _parameters(cx, cy, nu)
    if not np.isfinite(safety) or not 0 < safety <= 1:
        raise ValueError("safety must be finite and in (0, 1]")
    speed = abs(cx) + abs(cy)
    advection = h / speed if speed else np.inf
    diffusion = h**2 / (8.0 * nu) if nu else np.inf
    return float(safety * min(advection, diffusion))


def _rhs(u: Field, h: float, cx: float, cy: float, nu: float) -> Field:
    xp, xm = np.roll(u, -1, axis=0), np.roll(u, 1, axis=0)
    yp, ym = np.roll(u, -1, axis=1), np.roll(u, 1, axis=1)
    return (
        -cx * (xp - xm) / (2.0 * h)
        - cy * (yp - ym) / (2.0 * h)
        + nu * ((xp - u) + (xm - u) + (yp - u) + (ym - u)) / h**2
    )


def spatial_operator(u: ArrayLike, cx: float, cy: float, nu: float) -> Field:
    """Evaluate -cx*Dx(u) - cy*Dy(u) + nu*Laplacian(u), without mutation."""
    field = _field(u)
    _parameters(cx, cy, nu)
    return _rhs(field, _spacing(field.shape[0]), cx, cy, nu)


def _rk4(u: Field, dt: float, h: float, cx: float, cy: float, nu: float) -> Field:
    k1 = _rhs(u, h, cx, cy, nu)
    k2 = _rhs(u + 0.5 * dt * k1, h, cx, cy, nu)
    k3 = _rhs(u + 0.5 * dt * k2, h, cx, cy, nu)
    k4 = _rhs(u + dt * k3, h, cx, cy, nu)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _validate_dt(dt: float, limit: float) -> None:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if dt > limit:
        raise ValueError(f"dt exceeds the conservative RK4 stability bound {limit:g}")


def rk4_step(u: ArrayLike, dt: float, cx: float, cy: float, nu: float) -> Field:
    """Advance one classical RK4 step; validate dt and leave u unchanged."""
    field = _field(u)
    n = field.shape[0]
    _validate_dt(dt, stable_timestep(n, cx, cy, nu, safety=1.0))
    return _rk4(field, dt, _spacing(n), cx, cy, nu)


def solve(
    u0: ArrayLike, t_final: float, cx: float, cy: float, nu: float,
    dt: float | None = None,
) -> Field:
    """Integrate from time zero to t_final and return only the final field.

    If dt is omitted, use stable_timestep with its default safety factor.
    A supplied dt must satisfy the conservative bound. The last step is
    shortened to reach t_final. The input is never modified, including for
    zero duration or zero coefficients.
    """
    u = _field(u0).copy()
    n = u.shape[0]
    limit = stable_timestep(n, cx, cy, nu, safety=1.0)
    if not np.isfinite(t_final) or t_final < 0:
        raise ValueError("t_final must be finite and nonnegative")
    if dt is not None:
        _validate_dt(dt, limit)
    if t_final == 0 or np.isinf(limit):
        return u
    if dt is None:
        dt = stable_timestep(n, cx, cy, nu)
    _validate_dt(dt, limit)
    h = _spacing(n)
    t = 0.0
    while t < t_final:
        step = min(dt, t_final - t)
        if t + step == t:
            raise ValueError("dt is too small to advance time at floating-point precision")
        u = _rk4(u, step, h, cx, cy, nu)
        t = t_final if step == t_final - t else t + step
    return u
