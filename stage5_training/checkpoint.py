"""Candidate loading requires the already-frozen validation selection."""
import json
from pathlib import Path
import torch
from neural_operator.fno import FNO2d
from neural_operator.normalization import Normalization
from stage4_evaluation.checkpoint import verify_hash,APPROVED_HASH


def load_candidate(run,variant):
    if variant not in ('B','C'):
        raise ValueError('Expected variant B or C')
    run=Path(run)
    protocol=json.loads((run/'protocol.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    if selection['test_or_ood_used'] or not selection['all_gates_passed']:
        raise ValueError('Invalid training/evaluation separation')
    for path,digest in protocol['source_sha256'].items():
        verify_hash(path,digest)
    chosen=selection[variant]
    path=run/chosen['directory']/chosen['checkpoint']
    verify_hash(path,chosen['sha256'])
    state=torch.load(path,map_location='cpu',weights_only=True)
    for key in ('architecture','normalization','split_ids','config','initial_state_sha256'):
        if state[key]!=protocol[key]:
            raise ValueError(f'Checkpoint provenance mismatch: {key}')
    expected_lambda=0. if variant=='B' else chosen['lambda_mass']
    if state['epoch']!=chosen['epoch'] or state['lambda_mass']!=expected_lambda or state['validation']!=chosen['validation']:
        raise ValueError('Checkpoint selection mismatch')
    if protocol['historical_checkpoint_sha256']!=APPROVED_HASH:
        raise ValueError('Wrong historical baseline')
    model=FNO2d(**state['architecture'])
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.eval().requires_grad_(False)
    return model,Normalization(**state['normalization']),chosen
