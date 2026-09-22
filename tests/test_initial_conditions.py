"""Verify the periodized Gaussian, including behavior across the seam."""

import numpy as np
import pytest

from advection_diffusion import periodic_gaussian, periodic_grid


@pytest.mark.parametrize("sigma", [0.2, 1., 5.])
def test_gaussian_periodicity(sigma: float) -> None:
    x, y = periodic_grid(32)
    center = (0.05, 2 * np.pi - 0.1)
    u = periodic_gaussian(x, y, center, sigma)
    np.testing.assert_allclose(periodic_gaussian(x + 2 * np.pi, y - 4 * np.pi, center, sigma), u, atol=2e-14)
    np.testing.assert_allclose(periodic_gaussian(x, y, (center[0] + 2 * np.pi, center[1]), sigma), u, atol=2e-14)
    assert np.all(u >= 0)


def test_gaussian_mass_and_peak() -> None:
    x, y = periodic_grid(128)
    sigma, amplitude = 0.4, 2.
    u = periodic_gaussian(x, y, sigma=sigma, amplitude=amplitude)
    assert u[64, 64] == pytest.approx(amplitude)
    assert u.sum() * (2 * np.pi / 128)**2 == pytest.approx(amplitude * 2 * np.pi * sigma**2, rel=1e-12)


def test_smooth_seam_and_broadcasting() -> None:
    epsilon = 1e-5
    x = np.array([-epsilon, 0., epsilon, 2 * np.pi - epsilon, 2 * np.pi, 2 * np.pi + epsilon])
    values = periodic_gaussian(x, 0., center=(0.2, 0.), sigma=0.7)
    np.testing.assert_allclose(values[:3], values[3:], atol=1e-14)
    left = (values[1] - values[0]) / epsilon
    right = (values[2] - values[1]) / epsilon
    assert abs(left - right) < 3e-5


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf])
def test_invalid_width(sigma: float) -> None:
    with pytest.raises(ValueError):
        periodic_gaussian(0., 0., sigma=sigma)


@pytest.mark.parametrize("kwargs", [{"center": (0., np.nan)}, {"center": (0.,)}, {"amplitude": np.inf}])
def test_invalid_gaussian_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        periodic_gaussian(0., 0., **kwargs)
