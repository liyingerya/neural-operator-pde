"""Hash checks and exact reuse; no normalization fitting or split generation."""
import json
import subprocess
from pathlib import Path
import torch
from neural_operator.data import TrajectoryPairDataset, validate_baseline
from neural_operator.fno import FNO2d
from neural_operator.normalization import Normalization
from stage4_evaluation.checkpoint import load_frozen, verify_hash, sha256
from stage5_training.training import fresh_model, state_hash


def verify_snapshot(snapshot):
    for path,digest in snapshot.items():
        verify_hash(path,digest)
    return len(snapshot)


def protected_snapshot():
    paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())-{'README.md','pyproject.toml'}
    for root in ['data','runs/stage3_baseline','runs/stage4','runs/stage5']:
        paths.update(str(p) for p in Path(root).rglob('*') if p.is_file())
    paths.update(str(p) for p in Path('docs').glob('*.md') if 'learning' in p.name or 'walkthrough' in p.name)
    return {p:sha256(p) for p in sorted(paths)}


def frozen_context(config):
    historical,norm,baseline=load_frozen()
    protocol=json.loads(Path('runs/stage5/protocol.json').read_text())
    for key,value in [('architecture',historical.config),('normalization',norm.to_dict()),('split_ids',baseline['split_ids'])]:
        if protocol[key]!=value:
            raise ValueError(f'Frozen metadata mismatch: {key}')
    for path,digest in protocol['source_sha256'].items():
        verify_hash(path,digest)
    model=fresh_model(config,protocol['architecture'])
    if state_hash(model)!=protocol['initial_state_sha256']:
        raise ValueError('Initialization differs from Stage 5')
    return norm,baseline,protocol,model


def pairs(data,split_ids,normalization):
    validate_baseline(data)
    if set(split_ids)!= {'train','validation','test'}:
        raise ValueError('Expected the frozen three-way split')
    joined=sum((list(split_ids[k]) for k in ['train','validation','test']),[])
    if sorted(joined)!=list(range(64)) or [len(split_ids[k]) for k in ['train','validation','test']]!=[48,8,8]:
        raise ValueError('Trajectory leakage or invalid membership')
    return {key:TrajectoryPairDataset(data,ids,normalization) for key,ids in split_ids.items()}


def load_d(run):
    run=Path(run)
    selection=json.loads((run/'selection.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    if not selection['gate_passed'] or selection['test_or_ood_used']:
        raise ValueError('Selection must be validation-only after a passed gate')
    verify_hash(run/'protocol.json',selection['protocol_sha256'])
    verify_snapshot(protocol['source_sha256'])
    verify_hash(run/'best.pt',selection['sha256'])
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    for key in ['architecture','normalization','split_ids','initial_state_sha256','config']:
        if state[key]!=protocol[key]:
            raise ValueError(f'Checkpoint metadata mismatch: {key}')
    if state['epoch']!=selection['epoch'] or state['validation']!=selection['validation']:
        raise ValueError('Selected checkpoint mismatch')
    model=FNO2d(**state['architecture'])
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.eval().requires_grad_(False)
    return model,Normalization(**state['normalization']),selection
