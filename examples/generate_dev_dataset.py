"""Generate the small development archive: python -m examples.generate_dev_dataset."""

import argparse
from pathlib import Path

import numpy as np

from dataset_generation import GenerationConfig, generate_dataset, save_dataset


def main() -> None:
    """Generate the fixed 64-trajectory development dataset and print diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("data/dev/trajectories.npz"))
    args = parser.parse_args()
    config = GenerationConfig(seed=args.seed)
    data = generate_dataset(config)
    path = save_dataset(args.output, data)
    for key, value in data.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    for j, name in enumerate(("cx", "cy", "nu")):
        values = data["coefficients"][:, j]
        print(f"{name}: [{values.min():.9g}, {values.max():.9g}]")
    mask = data["blob_mask"]
    for key in ("blob_centers", "blob_widths", "blob_amplitudes"):
        values = data[key][mask]
        print(f"{key} active range: [{values.min():.9g}, {values.max():.9g}]")
    print(f"Blob count histogram (1,2,3): {np.bincount(data['blob_counts'], minlength=4)[1:].tolist()}")
    fields = data["fields"]
    mass = fields.sum(axis=(-2, -1)) * (2 * np.pi / config.grid_size)**2
    print(f"Field range: [{fields.min():.9g}, {fields.max():.9g}]")
    print(f"Maximum absolute mass drift: {np.max(np.abs(mass - mass[:, :1])):.3e}")
    print(f"Negative entries: {np.count_nonzero(fields < 0)} (not clipped)")
    print(f"Archive: {path.resolve()} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
