"""Pickle-free compressed NumPy archives with schema validation."""

import json
from pathlib import Path

import numpy as np

from advection_diffusion import stable_timestep
from .config import GenerationConfig
from .generate import B_MAX, COEFFICIENT_ORDER, FIELD_AXES, SCHEMA_VERSION


def validate_dataset(data: dict[str, np.ndarray]) -> dict:
    """Validate structure, finite values, masks, ranges, and metadata; return metadata.

    This does not recompute PDE trajectories. Dataset generation and the
    independent solver-consistency tests establish the numerical behavior.
    """
    keys = {"fields", "coefficients", "times", "trajectory_ids", "blob_counts",
            "blob_mask", "blob_centers", "blob_widths", "blob_amplitudes",
            "base_dt", "grid_size", "metadata_json"}
    if set(data) != keys or not all(isinstance(a, np.ndarray) for a in data.values()):
        raise ValueError("Dataset must contain exactly the schema arrays")
    encoded = data["metadata_json"]
    if encoded.shape != () or encoded.dtype.kind != "U":
        raise ValueError("metadata_json must be a scalar Unicode array")
    try:
        metadata = json.loads(encoded.item())
        config = GenerationConfig(**metadata["config"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Invalid generation metadata") from exc
    m, n, s = config.num_trajectories, config.grid_size, config.num_snapshots
    expected = {
        "fields": ((m, s, n, n), np.float64),
        "coefficients": ((m, 3), np.float64), "times": ((s,), np.float64),
        "trajectory_ids": ((m,), np.int64), "blob_counts": ((m,), np.int64),
        "blob_mask": ((m, B_MAX), np.bool_), "blob_centers": ((m, B_MAX, 2), np.float64),
        "blob_widths": ((m, B_MAX), np.float64), "blob_amplitudes": ((m, B_MAX), np.float64),
        "base_dt": ((m,), np.float64), "grid_size": ((), np.int64),
    }
    for name, (shape, dtype) in expected.items():
        a = data[name]
        if a.shape != shape or a.dtype != np.dtype(dtype) or not np.all(np.isfinite(a)):
            raise ValueError(f"Invalid shape, dtype, or nonfinite values in {name}")
    if data["grid_size"].item() != n:
        raise ValueError("grid_size disagrees with configuration")
    conventions = {"schema_version": SCHEMA_VERSION, "b_max": B_MAX,
                   "coefficient_order": COEFFICIENT_ORDER, "field_axes": FIELD_AXES,
                   "center_axes": ["x", "y"], "periodic": [True, True],
                   "endpoint_excluded": True, "rng_algorithm": "PCG64",
                   "time_integrator": "classical_RK4",
                   "spatial_scheme": "second_order_centered_np_roll",
                   "domain": [[0.0, float(2 * np.pi)], [0.0, float(2 * np.pi)]],
                   "grid_spacing": float(2 * np.pi / n),
                   "snapshot_schedule": "linspace_0_to_t_final_inclusive",
                   "interval_strategy": "sequential_solve_with_shortened_final_steps",
                   "base_dt_meaning": "nominal_RK4_step_before_interval_end_shortening",
                   "split_policy": "none_generated; defer_interpolation_and_OOD_splits_to_ML_stage"}
    if any(metadata.get(key) != value for key, value in conventions.items()):
        raise ValueError("Metadata conventions disagree with schema")
    if not np.array_equal(data["times"], config.snapshot_times()) or not np.all(np.diff(data["times"]) > 0):
        raise ValueError("Snapshot times disagree with configuration")
    if not np.array_equal(data["trajectory_ids"], np.arange(m)):
        raise ValueError("Trajectory IDs must be consecutive unique archive-local IDs")
    counts, mask = data["blob_counts"], data["blob_mask"]
    if np.any((counts < 1) | (counts > B_MAX)) or not np.array_equal(mask, np.arange(B_MAX)[None, :] < counts[:, None]):
        raise ValueError("blob_mask and blob_counts disagree")
    for name in ("blob_centers", "blob_widths", "blob_amplitudes"):
        if np.any(data[name][~mask] != 0):
            raise ValueError(f"Inactive slots in {name} must be zero")
    centers = data["blob_centers"][mask]
    if np.any((centers < 0) | (centers >= 2 * np.pi)):
        raise ValueError("Centers must lie in [0, 2*pi)")
    ranged = [("blob_widths", data["blob_widths"][mask], config.width_range),
              ("blob_amplitudes", data["blob_amplitudes"][mask], config.amplitude_range)]
    for j, name in enumerate(COEFFICIENT_ORDER):
        ranged.append((name, data["coefficients"][:, j], getattr(config, name + "_range")))
    for name, values, (low, high) in ranged:
        if np.any((values < low) | (values > high)):
            raise ValueError(f"{name} outside configured range")
    expected_dt = np.array([stable_timestep(n, *row, safety=config.safety)
                            for row in data["coefficients"]])
    if np.any(data["base_dt"] <= 0) or not np.allclose(data["base_dt"], expected_dt, rtol=1e-14, atol=0):
        raise ValueError("base_dt disagrees with coefficients and safety factor")
    return metadata


def save_dataset(path: str | Path, data: dict[str, np.ndarray]) -> Path:
    """Validate and save one compressed .npz archive; refuse accidental overwrite."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Dataset path must end in .npz")
    validate_dataset(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez_compressed(output, **data)
    return path


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load without pickle, validate, and return independent in-memory arrays."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_dataset(data)
    return data
