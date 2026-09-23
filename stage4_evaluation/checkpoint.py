"""Validate the exact approved Stage 3 checkpoint and its provenance."""
import hashlib
import json
from pathlib import Path
import torch
from neural_operator.fno import FNO2d
from neural_operator.normalization import Normalization

APPROVED_HASH = '300f7c54920c3d5d867fca1e059f7cd1024bc46499fc3c2a15a4b63c8b322f87'
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Provenance hash mismatch: {path}')


def load_frozen(run_dir=ROOT/'runs/stage3_baseline', archive=ROOT/'data/dev/trajectories.npz'):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/'manifest.json').read_text())
    selected = manifest['selected_checkpoint']
    if selected['epoch'] != 31 or selected['sha256'] != APPROVED_HASH:
        raise ValueError('Expected the approved epoch-31 checkpoint')
    checkpoint_path = run_dir/selected['path']
    verify_hash(checkpoint_path, APPROVED_HASH)
    verify_hash(archive, manifest['archive_sha256'])
    for recorded_path, digest in manifest['source_sha256'].items():
        path = Path(recorded_path)
        # The historical manifest recorded an absolute training-script path.
        path = ROOT/'examples'/path.name if path.is_absolute() else ROOT/path
        verify_hash(path, digest)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if state['epoch'] != 31 or state['normalization'] != manifest['normalization'] or state['split_ids'] != manifest['split_ids']:
        raise ValueError('Checkpoint and manifest disagree')
    if any(manifest['architecture'][key] != value for key, value in state['architecture'].items()):
        raise ValueError('Checkpoint architecture disagrees with manifest')
    model = FNO2d(**state['architecture'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model, Normalization(**state['normalization']), manifest


def write_json(path, data):
    """Exclusive output creation prevents silent replacement of measured results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
