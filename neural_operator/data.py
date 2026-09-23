"""Trajectory-first splits and lazy consecutive-snapshot pairs."""
import numpy as np
import torch
from torch.utils.data import Dataset


def trajectory_split(count=64, seed=2027):
    if count != 64:
        raise ValueError("Stage 3 baseline requires 64 trajectories")
    ids = np.random.Generator(np.random.PCG64(seed)).permutation(count)
    return {name: values.tolist() for name, values in
            zip(('train', 'validation', 'test'), (ids[:48], ids[48:56], ids[56:]))}


def validate_baseline(data):
    if data['fields'].shape != (64, 11, 64, 64):
        raise ValueError("Expected development fields with shape (64,11,64,64)")
    if data['coefficients'].shape != (64, 3):
        raise ValueError("Expected coefficients ordered cx, cy, nu")
    if data['times'].shape != (11,) or not np.allclose(
            np.diff(data['times']), 0.1, rtol=0, atol=1e-12):
        raise ValueError("Stage 3 requires a fixed 0.1 snapshot interval")


class TrajectoryPairDataset(Dataset):
    def __init__(self, data, trajectory_ids, normalization):
        ids = list(trajectory_ids)
        if not ids or len(set(ids)) != len(ids) or any(
                i < 0 or i >= len(data['fields']) for i in ids):
            raise ValueError("Invalid trajectory IDs")
        self.data = data
        self.normalization = normalization
        self.pairs = [(i, k) for i in ids for k in range(data['fields'].shape[1] - 1)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        i, k = self.pairs[index]
        norm = self.normalization
        field = torch.tensor(norm.field(self.data['fields'][i, k]), dtype=torch.float32)
        coefficients = torch.tensor(norm.coefficients(self.data['coefficients'][i]),
                                    dtype=torch.float32)
        channels = coefficients[:, None, None].expand(3, *field.shape)
        inputs = torch.cat((field[None], channels), dim=0)
        target = torch.tensor(norm.field(self.data['fields'][i, k+1]), dtype=torch.float32)[None]
        return inputs, target, i, k
