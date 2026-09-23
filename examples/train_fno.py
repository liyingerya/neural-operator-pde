"""Run the Stage 3 CPU baseline: python -m examples.train_fno."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from dataset_generation import load_dataset
from neural_operator.data import TrajectoryPairDataset, trajectory_split, validate_baseline
from neural_operator.normalization import Normalization
from neural_operator.fno import FNO2d, parameter_counts
from neural_operator.training import seed_everything, make_loader, overfit_gate, train_epoch, evaluate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/dev/trajectories.npz'))
    parser.add_argument('--output', type=Path, default=Path('runs/stage3_baseline'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error('epochs and threads must be positive')
    data = load_dataset(args.archive)
    validate_baseline(data)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    splits = trajectory_split()
    normalization = Normalization.fit(data, splits['train'])
    datasets = {name: TrajectoryPairDataset(data, ids, normalization) for name, ids in splits.items()}
    seed_everything(2028, args.threads)
    model = FNO2d()
    source_paths = sorted(Path('neural_operator').glob('*.py')) + [Path(__file__)]
    manifest = {
        'status': 'overfit_gate_running', 'archive': str(args.archive.resolve()),
        'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        'architecture': {**model.config, 'input_channels': 4, 'output_channels': 1,
                         'coordinates': False, 'activation': 'GELU'},
        **parameter_counts(model), 'split_seed': 2027, 'split_ids': splits,
        'normalization': normalization.to_dict(), 'training_seed': 2028,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'torch': torch.__version__},
        'device': 'cpu', 'platform': platform.platform(), 'threads': args.threads,
        'dtype': 'float32/complex64', 'deterministic_algorithms': True,
        'training': {'optimizer': 'Adam', 'learning_rate': 1e-3, 'weight_decay': 0,
                     'batch_size': 8, 'epochs': args.epochs, 'scheduler': None},
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'selected_checkpoint': None,
    }
    write_json(args.output/'manifest.json', manifest)
    gate = overfit_gate(model, datasets['train'], normalization)
    write_json(args.output/'overfit.json', gate)
    manifest['overfit_gate'] = {key: gate[key] for key in ('passed', 'epochs', 'criteria', 'runtime_seconds')}
    if not gate['passed']:
        manifest['status'] = 'stopped_overfit_gate_failed'
        write_json(args.output/'manifest.json', manifest)
        raise SystemExit('Overfit gate failed; full training was not started. See overfit.json.')
    # Discard gate-trained weights and reset both initialization and shuffle seeds.
    seed_everything(2028, args.threads)
    model = FNO2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = make_loader(datasets['train'], shuffle=True)
    val_loader = make_loader(datasets['validation'])
    best = float('inf')
    history = []
    training_start = time.perf_counter()
    manifest['status'] = 'training'
    write_json(args.output/'manifest.json', manifest)
    for epoch in range(1, args.epochs+1):
        epoch_start = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer)
        validation = evaluate(model, val_loader, normalization)
        history.append({'epoch': epoch, 'train_normalized_mse': loss,
                        'validation_normalized_mse': validation['normalized_mse'],
                        'validation_relative_l2': validation['relative_l2'],
                        'validation_relative_mass_error': validation['relative_mass_error'],
                        'runtime_seconds': time.perf_counter()-epoch_start})
        if validation['relative_l2'] < best:
            best = validation['relative_l2']
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                          'architecture': model.config, 'normalization': normalization.to_dict(),
                          'split_ids': splits, 'training_seed': 2028}
            torch.save(checkpoint, args.output/'best.pt')
            manifest['selected_checkpoint'] = {'path': 'best.pt', 'epoch': epoch,
                                               'validation_relative_l2': best}
        write_json(args.output/'history.json', history)
        print(f"epoch {epoch}: train_mse={loss:.6g} val_l2={validation['relative_l2']:.6g} "
              f"val_mass={validation['relative_mass_error']:.6g} "
              f"seconds={history[-1]['runtime_seconds']:.2f}", flush=True)
    manifest['training_runtime_seconds'] = time.perf_counter()-training_start
    checkpoint = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loader = make_loader(datasets['test'])
    results = {'validation': evaluate(model, val_loader, normalization),
               'test': evaluate(model, test_loader, normalization),
               'persistence_test': evaluate(model, test_loader, normalization, persistence=True)}
    write_json(args.output/'evaluation.json', results)
    manifest['selected_checkpoint']['sha256'] = hashlib.sha256((args.output/'best.pt').read_bytes()).hexdigest()
    manifest['total_runtime_seconds'] = time.perf_counter()-start
    manifest['status'] = 'complete'
    write_json(args.output/'manifest.json', manifest)
    print(json.dumps({name: {k: v for k, v in result.items() if k not in ('pairs', 'per_trajectory')}
                      for name, result in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
