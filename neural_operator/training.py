"""Deterministic CPU training, hard overfit gate, and physical evaluation."""
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .metrics import aggregate, physical_errors


def seed_everything(seed=2028, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def make_loader(dataset, shuffle=False, seed=2028, batch_size=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(seed))


def train_epoch(model, loader, optimizer):
    model.train()
    total, count = 0.0, 0
    for inputs, target, _, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = torch.mean((prediction-target)**2)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        total += loss.item() * len(inputs)
        count += len(inputs)
    return total/count


@torch.no_grad()
def evaluate(model, loader, normalization, persistence=False):
    model.eval()
    records, pair_records = [], []
    total, count = 0.0, 0
    for inputs, target, ids, steps in loader:
        prediction = inputs[:, :1] if persistence else model(inputs)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite prediction')
        total += torch.sum((prediction-target)**2).item()
        count += target.numel()
        physical_prediction = normalization.inverse_field(prediction.double())
        physical_target = normalization.inverse_field(target.double())
        l2, mass = physical_errors(physical_prediction, physical_target)
        for i, k, error, drift in zip(ids.tolist(), steps.tolist(), l2.tolist(), mass.tolist()):
            records.append((i, error, drift))
            pair_records.append({'trajectory_id': i, 'time_index': k,
                                 'relative_l2': error, 'relative_mass_error': drift})
    result = aggregate(records)
    result.update(normalized_mse=total/count, pairs=pair_records)
    return result


def overfit_passed(initial, final):
    """Predeclared gate: >=95% MSE reduction AND <=5% physical relative L2."""
    return bool(np.isfinite(final['normalized_mse']) and np.isfinite(final['relative_l2'])
                and final['normalized_mse'] <= 0.05 * initial['normalized_mse']
                and final['relative_l2'] <= 0.05)


def overfit_gate(model, training_data, normalization, epochs=300):
    # One initial-time pair from each of the first eight training trajectories.
    stride = training_data.data['fields'].shape[1]-1
    indices = [i*stride for i in range(8)]
    subset = Subset(training_data, indices)
    loader = make_loader(subset)
    initial = evaluate(model, loader, normalization)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        train_epoch(model, loader, optimizer)
        if epoch % 25 == 0:
            current = evaluate(model, loader, normalization)
            print(f"gate epoch {epoch}: mse={current['normalized_mse']:.6g}, "
                  f"physical_l2={current['relative_l2']:.6g}", flush=True)
    final = evaluate(model, loader, normalization)
    return {'passed': overfit_passed(initial, final), 'epochs': epochs,
            'criteria': {'maximum_mse_ratio': 0.05, 'maximum_physical_relative_l2': 0.05},
            'pairs': [list(training_data.pairs[i]) for i in indices],
            'initial': initial, 'final': final, 'runtime_seconds': time.perf_counter()-start}
