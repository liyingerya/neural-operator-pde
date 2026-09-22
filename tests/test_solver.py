"""Conservation, stability, and convergence verification."""

import numpy as np
import pytest

from advection_diffusion import periodic_grid, rk4_step, solve, spatial_operator, stable_timestep


def rms(u: np.ndarray) -> float:
    """Return the root-mean-square norm."""
    return float(np.sqrt(np.mean(u**2)))


def test_grid() -> None:
    x, y = periodic_grid(16)
    assert x.shape == y.shape == (16, 16)
    np.testing.assert_allclose(x[:, 0], np.arange(16) * 2 * np.pi / 16)
    np.testing.assert_array_equal(x, y.T)
    assert x.min() == 0 and x.max() < 2 * np.pi


@pytest.mark.parametrize("nu", [0.0, 1e-8, 0.1])
def test_constant_preservation(nu: float) -> None:
    u = np.full((16, 16), 2.75)
    np.testing.assert_array_equal(spatial_operator(u, 0.7, -0.4, nu), 0)
    np.testing.assert_array_equal(solve(u, 0.5, 0.7, -0.4, nu), u)


@pytest.mark.parametrize("nu", [0.0, 1e-8, 0.1])
def test_mass_conservation_and_no_mutation(nu: float) -> None:
    u = np.random.default_rng(42).normal(size=(24, 24)) + 1
    original = u.copy()
    final = solve(u, 1.0, 0.8, -0.3, nu)
    np.testing.assert_allclose(final.sum(), u.sum(), rtol=0, atol=2e-12)
    np.testing.assert_array_equal(u, original)
    assert not np.shares_memory(u, final)


@pytest.mark.parametrize("cx,cy,nu", [(0.7, -0.4, 0.), (0.7, -0.4, 1e-8), (0.7, -0.4, 0.08), (0., 0., 0.08)])
def test_fourier_mode_and_spatial_convergence(cx: float, cy: float, nu: float) -> None:
    errors = []
    t = 0.3
    for n in (16, 32, 64):
        x, y = periodic_grid(n)
        phase = 2 * x + y
        exact = np.exp(-5 * nu * t) * np.cos(phase - (2 * cx + cy) * t)
        final = solve(np.cos(phase), t, cx, cy, nu, dt=min(0.002, stable_timestep(n, cx, cy, nu)))
        errors.append(rms(final - exact))
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 3.7) & (ratios < 4.3)), (errors, ratios)
    assert errors[-1] < 0.002


@pytest.mark.parametrize("cx,cy,nu", [(1., 0., 0.), (0., 1., 0.), (0., 0., 0.2)])
def test_spatial_operator_convergence(cx: float, cy: float, nu: float) -> None:
    errors = []
    for n in (16, 32, 64):
        x, y = periodic_grid(n)
        phase = 2 * x + 3 * y
        exact = (2 * cx + 3 * cy) * np.sin(phase) - 13 * nu * np.cos(phase)
        errors.append(rms(spatial_operator(np.cos(phase), cx, cy, nu) - exact))
    assert np.all(np.array(errors[:-1]) / errors[1:] > 3.7)


@pytest.mark.parametrize("nu", [0., 0.01])
def test_rk4_temporal_order(nu: float) -> None:
    n, cx, cy, t = 16, 1., -0.4, 0.4
    x, y = periodic_grid(n)
    h = 2 * np.pi / n
    phase = 3 * x + 2 * y
    a = 4 * nu / h**2 * (np.sin(3 * h / 2)**2 + np.sin(h)**2)
    b = (cx * np.sin(3 * h) + cy * np.sin(2 * h)) / h
    exact = np.exp(-a * t) * np.cos(phase - b * t)
    errors = [rms(solve(np.cos(phase), t, cx, cy, nu, dt) - exact)
              for dt in (0.1, 0.05, 0.025)]
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 14) & (ratios < 18)), (errors, ratios)


def test_shortened_final_step() -> None:
    x, y = periodic_grid(16)
    u = np.cos(x + y)
    expected = u.copy()
    for step in (0.1, 0.1, 0.035):
        expected = rk4_step(expected, step, 0.5, -0.2, 0.01)
    np.testing.assert_allclose(solve(u, 0.235, 0.5, -0.2, 0.01, 0.1), expected, atol=1e-15)
    np.testing.assert_array_equal(u, np.cos(x + y))


def test_zero_duration_and_stationary_equation() -> None:
    u = np.random.default_rng(0).normal(size=(8, 8))
    for final in (solve(u, 0, 1., 1., 0.1), solve(u, 100, 0., 0., 0.)):
        np.testing.assert_array_equal(final, u)
        assert not np.shares_memory(final, u)
    assert np.isinf(stable_timestep(8, 0, 0, 0))


def test_timestep_helper() -> None:
    h = 2 * np.pi / 16
    assert stable_timestep(16, 1., -2., 0.) == pytest.approx(0.9 * h / 3)
    assert stable_timestep(16, 0., 0., 0.1) == pytest.approx(0.9 * h**2 / 0.8)
    assert stable_timestep(16, 1., -2., 0.1) == pytest.approx(0.9 * min(h / 3, h**2 / 0.8))


@pytest.mark.parametrize("cx,cy,nu", [(1., -0.4, 0.), (1., -0.4, 1e-8), (1., -0.4, 0.1), (0., 0., 0.1)])
def test_all_discrete_fourier_modes_stable(cx: float, cy: float, nu: float) -> None:
    n = 32
    h = 2 * np.pi / n
    theta_x, theta_y = periodic_grid(n)
    eigenvalues = (-4 * nu / h**2 * (np.sin(theta_x / 2)**2 + np.sin(theta_y / 2)**2)
                  - 1j * (cx * np.sin(theta_x) + cy * np.sin(theta_y)) / h)
    z = stable_timestep(n, cx, cy, nu, safety=1) * eigenvalues
    amplification = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    assert np.max(np.abs(amplification)) <= 1 + 1e-14


@pytest.mark.parametrize("n", [0, 2, 3.5, True])
def test_invalid_grid(n: int) -> None:
    with pytest.raises(ValueError):
        periodic_grid(n)


@pytest.mark.parametrize("safety", [0, -0.1, 1.1, np.nan])
def test_invalid_safety(safety: float) -> None:
    with pytest.raises(ValueError):
        stable_timestep(16, 1., 0., 0., safety)


@pytest.mark.parametrize("dt", [0., -1., np.nan, np.inf, 1.])
def test_invalid_timestep(dt: float) -> None:
    with pytest.raises(ValueError):
        solve(np.ones((16, 16)), 1., 1., 0., 0.1, dt)
    with pytest.raises(ValueError):
        rk4_step(np.ones((16, 16)), dt, 1., 0., 0.1)


@pytest.mark.parametrize("u", [np.ones(4), np.ones((4, 5)), np.ones((2, 2)), np.full((4, 4), np.nan), np.ones((4, 4), dtype=complex)])
def test_invalid_field(u: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solve(u, 1., 1., 0., 0.1)


@pytest.mark.parametrize("cx,cy,nu", [(np.nan, 0., 0.), (0., np.inf, 0.), (0., 0., -0.1), (0., 0., np.nan)])
def test_invalid_coefficients(cx: float, cy: float, nu: float) -> None:
    with pytest.raises(ValueError):
        solve(np.ones((8, 8)), 1., cx, cy, nu)


@pytest.mark.parametrize("t_final", [-1., np.nan, np.inf])
def test_invalid_final_time(t_final: float) -> None:
    with pytest.raises(ValueError):
        solve(np.ones((8, 8)), t_final, 1., 0., 0.)
