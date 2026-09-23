# Stage 2: reproducible development trajectories

Stage 2 builds a small NumPy dataset using the unchanged Stage 1 solver as
the numerical source of truth. It adds no neural-network framework,
training code, split assignments, or production-generation pipeline.

## Generate and load

From the repository root, using the existing environment:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m examples.generate_dev_dataset
```

The default output is `data/dev/trajectories.npz`. The command generates
64 trajectories on a 64×64 grid, with 11 snapshots at equally spaced
physical times from 0 to 1, and seed 2026. Its only CLI option is `--output`;
use a new output path to rerun without replacing an archive. Saving refuses
to overwrite existing files. `/data/` is ignored by Git.

```python
from dataset_generation import load_dataset

archive = load_dataset("data/dev/trajectories.npz")
u0 = archive["fields"][:, 0]       # (64, 64, 64): trajectory, x, y
uT = archive["fields"][:, -1]      # same shape
parameters = archive["coefficients"]  # (64, 3): cx, cy, nu
```

`load_dataset` uses `allow_pickle=False` and validates shapes, dtypes,
finite values, ranges, masks, times, and metadata consistency. It loads
arrays into memory; the compressed format is not a memory-mapped store.
The validator does not rerun the PDE or authenticate the field contents.

The public `GenerationConfig` allows small configurations for tests and
experiments, but this Stage 2 run creates only the agreed development
archive. For example, `GenerationConfig(num_trajectories=4, grid_size=16,
num_snapshots=4, t_final=0.23, seed=42)` is the test fixture. Its coarse grid
is for API/consistency verification, not a recommended scientific dataset.

## Source files

| File | Responsibility |
| --- | --- |
| `dataset_generation/config.py` | Frozen configuration, parameter validation, uniform snapshot schedule |
| `dataset_generation/generate.py` | Local seeded RNG, blob sampling, Stage 1 interval solves, metadata/provenance |
| `dataset_generation/io.py` | Schema validation and compressed, pickle-free save/load |
| `dataset_generation/__init__.py` | Public imports |
| `examples/generate_dev_dataset.py` | Small default run and numerical diagnostics |
| `tests/test_dataset_generation.py` | Reproducibility, numerical consistency, schema, ranges, and error handling |

`pyproject.toml` includes the new package with no new numerical dependency.
Python standard-library modules handle JSON, filesystem paths, hashing, and
optional Git revision lookup. Numerical operations use NumPy and Stage 1.

## Exact archive schema

Let M be trajectory count, S snapshot count, N grid size, and B_max=3.
The archive contains exactly these arrays:

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `fields` | `(M, S, N, N)` | float64 | Trajectories; axes: trajectory, time, x, y |
| `coefficients` | `(M, 3)` | float64 | Columns: cx, cy, nu; fixed along a trajectory |
| `times` | `(S,)` | float64 | Shared physical times, including zero and T |
| `trajectory_ids` | `(M,)` | int64 | Consecutive archive-local IDs, zero through M-1 |
| `blob_counts` | `(M,)` | int64 | Active blob counts, 1–3 |
| `blob_mask` | `(M, 3)` | bool | True for active Gaussian slots |
| `blob_centers` | `(M, 3, 2)` | float64 | Last axis: x, y |
| `blob_widths` | `(M, 3)` | float64 | Gaussian standard deviations |
| `blob_amplitudes` | `(M, 3)` | float64 | Multipliers of each periodized image sum |
| `base_dt` | `(M,)` | float64 | Nominal RK4 timestep before interval-end shortening |
| `grid_size` | `()` | int64 | N |
| `metadata_json` | `()` | Unicode | JSON text, not a pickled Python object |

For trajectory i, slots `0:blob_counts[i]` are active. The mask is exactly
`arange(3) < blob_counts[i]`. Inactive centers, widths, and amplitudes are
zero. Use the mask when computing width/amplitude distributions; padding
zeros are not sampled blobs. All numerical entries, including padding,
are finite.

Coordinates are reconstructed with `advection_diffusion.periodic_grid(N)`.
The domain is `[0, 2π)²`, endpoint excluded, with x along array axis 0 and
y along axis 1 within each field.

### Metadata

Decode with `json.loads(archive["metadata_json"].item())`. It includes:

- `schema_version` (1), `generator_version` (0.1.0), and `numpy_version`.
- `rng_algorithm` (`PCG64`) and complete `config`, including the seed,
  dimensions, final time, all distribution bounds, and timestep safety.
- `domain`, `periodic`, `endpoint_excluded`, and `grid_spacing`.
- `field_axes`, `coefficient_order`, `center_axes`, and `b_max`.
- `sampling`, describing discrete blob counts, uniform center sampling,
  uniform continuous parameters, and inactive-slot padding.
- `time_integrator`, `spatial_scheme`, `snapshot_schedule`,
  `interval_strategy`, and `base_dt_meaning`.
- `split_policy`, explicitly stating that no splits are generated.
- `provenance`: available `git_revision` plus SHA-256 hashes of the actual
  Stage 1 and Stage 2 package source files.

The revision can be null outside a Git checkout. A revision alone does not
identify uncommitted work, so hashes are included. They record the source
used, not a guarantee that it has been committed. Config metadata is enough
to reconstruct the generation invocation. No wall-clock timestamp is used,
so repeated calls with the same code/configuration give identical metadata.

## Sampling and reproducibility

The default distributions are independent continuous uniforms except for
the discrete blob count:

| Parameter | Default distribution |
| --- | --- |
| Blob count | Uniform integer in {1, 2, 3} |
| Each x/y center | Uniform on [0, 2π) |
| Width | Uniform on [0.4, 0.8] |
| Amplitude | Uniform on [0.5, 1.5] |
| cx, cy | Each uniform on [-1, 1] |
| nu | Uniform on [0.005, 0.05] |

NumPy's uniform sampler normally excludes the upper bound; the range
validator allows equality at the upper parameter bound to accommodate
floating-point rounding. Centers are validated strictly below 2π.

Each initial condition sums 1–3 calls to Stage 1 `periodic_gaussian`.
Widths correspond to about 4–8 cells per standard deviation on the default
grid. Amplitudes multiply image sums rather than normalizing total mass;
overlapping blobs may have peaks above 1.5. No normalization is applied.
Pure advection and very weak diffusion remain Stage 1 verification cases;
the default development archive samples only the stated positive nu range.

The generator constructs `Generator(PCG64(seed))` locally. It does not seed
or consume NumPy's global RNG. Blob counts are sampled first, followed by
per-trajectory centers, widths, amplitudes, and coefficients. Reproducibility
assumes the same configuration, code, and numerical environment. Changing M
can change earlier trajectories because it changes the initial RNG draws;
this implementation does not promise prefix stability or parallel scheduling
independence. Identical arrays are promised within a matching environment,
not identical compressed ZIP bytes or cross-version bitwise identity.

## Trajectory integration and learning pairs

For each trajectory, compute `base_dt = stable_timestep(N, cx, cy, nu,
safety)` once (default safety=0.9). Store the initial field as snapshot zero.
For each later snapshot k, call:

```python
state = solve(
    state,
    t_final=float(times[k] - times[k - 1]),
    cx=cx, cy=cy, nu=nu, dt=base_dt,
)
```

The Stage 1 PDE is autonomous, so restarting its local clock at zero is
valid. `base_dt` is not the snapshot spacing, nor necessarily the actual
last internal step. Stage 1 shortens the final RK4 step in every interval
to land at that snapshot's physical time. There may be multiple internal
steps per snapshot interval, or just one shortened step.

An interval trajectory need not be bitwise identical to one uninterrupted
solve from zero to T, since intermediate boundaries change the RK4 step
partition. Tests compare with the same sequence of interval solves and
independently reconstruct initial conditions from stored blob parameters.

The archive can support later direct mappings `fields[:, 0] -> fields[:, -1]`
and consecutive-snapshot mappings `fields[:, k] -> fields[:, k+1]`. It stores
neither extracted pairs nor training splits now. Because coefficients vary,
a future model must receive cx, cy, nu and the requested time interval as
appropriate; a field alone does not determine its evolution across this
ensemble.

## Future split policy: documented, not implemented

No train/validation/test assignments or split seed are generated or stored
in Stage 2. Interpolation and out-of-distribution (OOD) definitions will be
designed at the ML stage rather than fixed by this development archive.

When splitting is introduced:

1. Split whole trajectories before extracting any direct or autoregressive
   pairs. All snapshots/pairs from a trajectory remain together.
2. If initial conditions are reused across coefficients or resolutions,
   group by initial-condition family before splitting to prevent leakage.
3. Fit normalization statistics on training data only.
4. Persist the eventual split configuration and assignments for reproducible
   experiments. IDs here are unique only within an archive.
5. Distinguish random in-distribution interpolation from explicit holdouts
   in diffusivity, velocity, width, resolution, or other physical regimes.

This 64-trajectory archive is for development, not a statistically reliable
benchmark. Adjacent snapshots are correlated observations, not independent
samples.

## Verification and numerical observations

The full suite passed: **90 cases** (60 unchanged Stage 1 cases and 30 new
Stage 2 cases). New tests check equal-seed equality, different-seed changes,
isolation from global RNG, shapes/dtypes/finiteness, ranges and padding,
metadata conventions, initial-condition reconstruction, every interval's
agreement with Stage 1, mass conservation, pickle-free round trips,
overwrite protection, and invalid configurations/corrupted schema arrays.

The development run used seed 2026 and produced `fields.shape =
(64, 11, 64, 64)`. The field array alone occupies 23,068,672 bytes (22 MiB).
The compressed archive occupies 21,993,361 bytes (approximately 20.97 MiB).
Compression depends on numerical content and the ZIP environment.

| Measured quantity | Minimum | Maximum |
| --- | ---: | ---: |
| cx | -0.996395293 | 0.973680553 |
| cy | -0.979536196 | 0.999566612 |
| nu | 0.00568725958 | 0.0497861944 |
| Active centers (both coordinates pooled) | 0.0270188953 | 6.27547553 |
| Active widths | 0.408506756 | 0.795894467 |
| Active amplitudes | 0.509210658 | 1.49943647 |
| All field entries | 1.66176525e-25 | 1.96991621 |

Blob counts 1, 2, and 3 occurred 21, 25, and 18 times, respectively.
Maximum absolute mass drift over all snapshots/trajectories was
`1.776e-15`. All field values were finite, with zero negative entries in
this sample. No clipping was applied. A repeated invocation at the same
output path was rejected by the overwrite safeguard; only one development
archive was saved.

These results do not imply a general positivity guarantee. Centered
advection and RK4 can produce small negative undershoots for other
parameters or resolutions. The generator preserves such finite values;
it raises on nonfinite solutions. It does not enforce an analytic-error
tolerance. Wider coverage, long rollouts, and resolution studies belong to
subsequent scientific validation, not an implicit guarantee of this archive.
