"""Stage 2 schema, reproducibility, and independent Stage 1 consistency checks."""

from dataclasses import replace
import json

import numpy as np
import pytest

from advection_diffusion import periodic_gaussian, periodic_grid, solve, stable_timestep
from dataset_generation import (GenerationConfig, generate_dataset, load_dataset,
                                save_dataset, validate_dataset)


@pytest.fixture(scope="module")
def config() -> GenerationConfig:
    """Keep verification much smaller than the development dataset."""
    return GenerationConfig(num_trajectories=4, grid_size=16, num_snapshots=4,
                            t_final=0.23, seed=42)


@pytest.fixture(scope="module")
def dataset(config: GenerationConfig) -> dict[str, np.ndarray]:
    """Generate one shared, read-only-by-convention fixture."""
    return generate_dataset(config)


def test_reproducible(config: GenerationConfig, dataset: dict) -> None:
    repeated = generate_dataset(config)
    for name in dataset:
        np.testing.assert_array_equal(dataset[name], repeated[name])


def test_different_seed(config: GenerationConfig, dataset: dict) -> None:
    other = generate_dataset(replace(config, seed=43))
    assert not np.array_equal(other["fields"], dataset["fields"])
    assert not np.array_equal(other["coefficients"], dataset["coefficients"])


def test_local_rng_does_not_touch_global(config: GenerationConfig) -> None:
    before = np.random.get_state()
    generate_dataset(config)
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_schema_shapes_metadata_and_finiteness(dataset: dict) -> None:
    assert dataset["fields"].shape == (4, 4, 16, 16)
    assert dataset["coefficients"].shape == (4, 3)
    assert dataset["times"].shape == (4,)
    assert dataset["blob_centers"].shape == (4, 3, 2)
    assert dataset["blob_widths"].shape == dataset["blob_amplitudes"].shape == (4, 3)
    assert dataset["blob_mask"].shape == (4, 3)
    assert dataset["blob_mask"].dtype == np.bool_
    assert dataset["blob_counts"].shape == dataset["base_dt"].shape == (4,)
    assert dataset["grid_size"].shape == dataset["metadata_json"].shape == ()
    for name, array in dataset.items():
        assert array.dtype.kind != "O"
        if name != "metadata_json":
            assert np.all(np.isfinite(array))
    metadata = validate_dataset(dataset)
    assert metadata["config"]["seed"] == 42
    assert metadata["coefficient_order"] == ["cx", "cy", "nu"]
    assert metadata["field_axes"] == ["trajectory", "time", "x", "y"]
    assert metadata["grid_spacing"] == 2 * np.pi / 16
    assert metadata["rng_algorithm"] == "PCG64"
    assert metadata["numpy_version"] == np.__version__
    assert "source_sha256" in metadata["provenance"]
    assert not any("split" in name for name in dataset)
    np.testing.assert_allclose(np.diff(dataset["times"]), 0.23 / 3)


def test_parameter_ranges_and_mask(dataset: dict, config: GenerationConfig) -> None:
    mask = dataset["blob_mask"]
    np.testing.assert_array_equal(mask.sum(axis=1), dataset["blob_counts"])
    assert np.all((dataset["blob_counts"] >= 1) & (dataset["blob_counts"] <= 3))
    for name in ("blob_centers", "blob_widths", "blob_amplitudes"):
        assert np.all(dataset[name][~mask] == 0)
    centers = dataset["blob_centers"][mask]
    assert np.all((centers >= 0) & (centers < 2 * np.pi))
    for name, bounds in (("blob_widths", config.width_range),
                         ("blob_amplitudes", config.amplitude_range)):
        assert np.all((dataset[name][mask] >= bounds[0]) & (dataset[name][mask] <= bounds[1]))
    for j, bounds in enumerate((config.cx_range, config.cy_range, config.nu_range)):
        assert np.all((dataset["coefficients"][:, j] >= bounds[0]) &
                      (dataset["coefficients"][:, j] <= bounds[1]))


def test_initial_reconstruction_and_solver_consistency(dataset: dict) -> None:
    x, y = periodic_grid(16)
    for i, (cx, cy, nu) in enumerate(dataset["coefficients"]):
        initial = np.zeros_like(x)
        for b in np.flatnonzero(dataset["blob_mask"][i]):
            initial += periodic_gaussian(x, y, tuple(dataset["blob_centers"][i, b]),
                                         dataset["blob_widths"][i, b],
                                         dataset["blob_amplitudes"][i, b])
        np.testing.assert_array_equal(initial, dataset["fields"][i, 0])
        assert dataset["base_dt"][i] == stable_timestep(16, cx, cy, nu)
        for k, duration in enumerate(np.diff(dataset["times"]), start=1):
            initial = solve(initial, float(duration), cx, cy, nu, dataset["base_dt"][i])
            np.testing.assert_array_equal(initial, dataset["fields"][i, k])
    masses = dataset["fields"].sum(axis=(-2, -1))
    np.testing.assert_allclose(masses, np.broadcast_to(masses[:, :1], masses.shape),
                               rtol=0, atol=1e-12)


def test_archive_round_trip(tmp_path, dataset: dict) -> None:
    path = save_dataset(tmp_path / "nested" / "dataset.npz", dataset)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(dataset)
    loaded = load_dataset(path)
    for name in dataset:
        np.testing.assert_array_equal(loaded[name], dataset[name])
    with pytest.raises(FileExistsError):
        save_dataset(path, dataset)
    with pytest.raises(ValueError):
        save_dataset(tmp_path / "wrong.extension", dataset)


@pytest.mark.parametrize("name", ["fields", "coefficients", "times", "grid_size",
                                  "blob_mask", "blob_counts", "blob_widths", "base_dt"])
def test_reject_inconsistent_archive(dataset: dict, name: str) -> None:
    corrupted = {key: value.copy() for key, value in dataset.items()}
    if name == "fields":
        corrupted[name].flat[0] = np.nan
    elif name == "coefficients":
        corrupted[name][0, 0] = 100
    elif name == "times":
        corrupted[name][-1] += 1
    elif name == "grid_size":
        corrupted[name][...] = 32
    elif name == "blob_mask":
        corrupted[name][0, 0] = False
    elif name == "blob_counts":
        corrupted[name][0] = 0
    elif name == "blob_widths":
        corrupted[name] = corrupted[name][:, :2]
    else:
        corrupted[name][0] *= 2
    with pytest.raises(ValueError):
        validate_dataset(corrupted)


def test_reject_metadata_disagreement(dataset: dict) -> None:
    corrupted = dataset.copy()
    metadata = json.loads(dataset["metadata_json"].item())
    metadata["coefficient_order"] = ["nu", "cx", "cy"]
    corrupted["metadata_json"] = np.asarray(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate_dataset(corrupted)


@pytest.mark.parametrize("kwargs", [
    {"num_trajectories": 0}, {"grid_size": 2}, {"num_snapshots": 1},
    {"seed": -1}, {"seed": True}, {"t_final": 0}, {"t_final": np.inf},
    {"safety": 0}, {"safety": 1.1}, {"width_range": (0, 0.8)},
    {"cx_range": (1, -1)}, {"cy_range": (0, np.nan)},
    {"nu_range": (-0.1, 0.1)}, {"amplitude_range": (0, 1)},
])
def test_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)
