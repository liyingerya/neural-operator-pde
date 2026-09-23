"""Sample smooth initial conditions and integrate using the Stage 1 API."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from advection_diffusion import periodic_gaussian, periodic_grid, solve, stable_timestep
from .config import GenerationConfig

B_MAX = 3
SCHEMA_VERSION = 1
GENERATOR_VERSION = "0.1.0"
COEFFICIENT_ORDER = ["cx", "cy", "nu"]
FIELD_AXES = ["trajectory", "time", "x", "y"]


def _provenance() -> dict:
    """Record available revision plus exact source hashes, including uncommitted code."""
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    paths = sorted((root / "advection_diffusion").glob("*.py"))
    paths += sorted((root / "dataset_generation").glob("*.py"))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    return {"git_revision": revision, "source_sha256": hashes}


def generate_dataset(config: GenerationConfig = GenerationConfig()) -> dict[str, np.ndarray]:
    """Generate float64 trajectories in memory, with reproducible PCG64 sampling.

    Each interval is a separate Stage 1 solve with the same base_dt. The
    solver may shorten the last internal step at every snapshot boundary.
    No normalization, clipping, split assignment, or file I/O is performed.
    """
    rng = np.random.Generator(np.random.PCG64(config.seed))
    m, n, s = config.num_trajectories, config.grid_size, config.num_snapshots
    times = config.snapshot_times()
    x, y = periodic_grid(n)
    fields = np.empty((m, s, n, n), dtype=np.float64)
    coefficients = np.empty((m, 3), dtype=np.float64)
    counts = rng.integers(1, B_MAX + 1, size=m, dtype=np.int64)
    mask = np.arange(B_MAX)[None, :] < counts[:, None]
    centers = np.zeros((m, B_MAX, 2), dtype=np.float64)
    widths = np.zeros((m, B_MAX), dtype=np.float64)
    amplitudes = np.zeros_like(widths)
    base_dt = np.empty(m, dtype=np.float64)
    for i in range(m):
        count = counts[i]
        centers[i, :count] = rng.uniform(0.0, 2.0 * np.pi, size=(count, 2))
        widths[i, :count] = rng.uniform(*config.width_range, size=count)
        amplitudes[i, :count] = rng.uniform(*config.amplitude_range, size=count)
        coefficients[i] = [rng.uniform(*bounds) for bounds in
                           (config.cx_range, config.cy_range, config.nu_range)]
        cx, cy, nu = coefficients[i]
        base_dt[i] = stable_timestep(n, cx, cy, nu, config.safety)
        state = np.zeros((n, n), dtype=np.float64)
        for b in range(count):
            state += periodic_gaussian(x, y, tuple(centers[i, b]),
                                       widths[i, b], amplitudes[i, b])
        fields[i, 0] = state
        for k in range(1, s):
            state = solve(state, float(times[k] - times[k - 1]), cx, cy, nu, base_dt[i])
            fields[i, k] = state
        if not np.all(np.isfinite(fields[i])):
            raise FloatingPointError(f"Nonfinite solution in trajectory {i}")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "rng_algorithm": "PCG64",
        "config": asdict(config),
        "domain": [[0.0, float(2 * np.pi)], [0.0, float(2 * np.pi)]],
        "periodic": [True, True],
        "endpoint_excluded": True,
        "grid_spacing": float(2 * np.pi / n),
        "field_axes": FIELD_AXES,
        "coefficient_order": COEFFICIENT_ORDER,
        "center_axes": ["x", "y"],
        "b_max": B_MAX,
        "sampling": {"blob_count": "discrete_uniform_1_to_3",
                     "centers": "independent_uniform_[0,2pi)",
                     "parameters": "independent_uniform_config_ranges",
                     "inactive_slots": "zero_padded"},
        "time_integrator": "classical_RK4",
        "spatial_scheme": "second_order_centered_np_roll",
        "snapshot_schedule": "linspace_0_to_t_final_inclusive",
        "interval_strategy": "sequential_solve_with_shortened_final_steps",
        "base_dt_meaning": "nominal_RK4_step_before_interval_end_shortening",
        "split_policy": "none_generated; defer_interpolation_and_OOD_splits_to_ML_stage",
        "provenance": _provenance(),
    }
    return {
        "fields": fields, "coefficients": coefficients, "times": times,
        "trajectory_ids": np.arange(m, dtype=np.int64), "blob_counts": counts,
        "blob_mask": mask, "blob_centers": centers, "blob_widths": widths,
        "blob_amplitudes": amplitudes, "base_dt": base_dt,
        "grid_size": np.asarray(n, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    }
