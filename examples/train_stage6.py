"""Train only the predeclared matched one-step control D."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from dataset_generation import load_dataset
from neural_operator.fno import parameter_counts
from stage4_evaluation.checkpoint import write_json,sha256
from stage5_training.training import fresh_model,state_hash,loader,selection_key
from stage6_control.config import Config,INTERPRETATION,budget
from stage6_control.provenance import frozen_context,pairs,protected_snapshot
from stage6_control.training import tiny_gate,train_epoch,validation_metrics


def save_progress(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def full_training(run,protocol,data,dataset,norm,config):
    model=fresh_model(config,protocol['architecture'])
    assert state_hash(model)==protocol['initial_state_sha256']
    batches=loader(dataset,config,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    history=[];best=None;selected=None;start=time.perf_counter();training_seconds=0.
    for epoch in range(1,config.epochs+1):
        tick=time.perf_counter()
        loss=train_epoch(model,batches,optimizer)
        train_seconds=time.perf_counter()-tick;training_seconds+=train_seconds
        validation=validation_metrics(model,data,protocol['split_ids']['validation'],norm)
        elapsed=time.perf_counter()-tick
        row={**budget(epoch),'training_loss':loss,'validation':validation,
             'training_seconds':train_seconds,'runtime_seconds':elapsed,
             'cumulative_training_seconds':training_seconds,
             'cumulative_epoch_seconds':sum(r['runtime_seconds'] for r in history)+elapsed,
             'elapsed_wall_seconds':time.perf_counter()-start}
        history.append(row)
        key=selection_key(validation,epoch)
        if best is None or key<best:
            best=key
            state={k:protocol[k] for k in ['architecture','normalization','split_ids','config','initial_state_sha256']}
            state.update(model_state_dict=model.state_dict(),epoch=epoch,validation=validation)
            torch.save(state,run/'best.pt')
            selected={'epoch':epoch,'validation':validation,'sha256':sha256(run/'best.pt'),'checkpoint':'best.pt'}
        save_progress(run/'history.json',history)
        print(f'D epoch={epoch}/60 MSE={loss:.6g} val_AR={validation["final_relative_l2"]:.6g} val_TF={validation["one_step_relative_l2"]:.6g} seconds={elapsed:.2f}',flush=True)
    runtime=time.perf_counter()-start
    selection={**selected,'gate_passed':True,'test_or_ood_used':False,'protocol_sha256':sha256(run/'protocol.json'),
               'runtime_seconds':runtime,'training_only_seconds':training_seconds,
               'completed_budget':budget(config.epochs),'selected_budget':budget(selected['epoch'])}
    write_json(run/'selection.json',selection)
    # E is optional; two minutes is the predeclared ceiling for a clearly modest addition.
    estimate=runtime/config.epochs*100
    write_json(run/'optional_e.json',{'estimated_seconds':estimate,'method':'D full-run mean epoch time times 100, excluding another gate',
        'clearly_modest_ceiling_seconds':120,'run_E':False,
        'reason':'Required D is complete; E is omitted if estimate exceeds the modest ceiling. No automatic E training is implemented.'})
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/stage6'))
    args=parser.parse_args();run=args.output;config=Config()
    if run.exists():
        raise FileExistsError('Refusing to overwrite a Stage 6 run')
    norm,baseline,previous,model=frozen_context(config)
    protected=protected_snapshot()
    data=load_dataset('data/dev/trajectories.npz')
    datasets=pairs(data,previous['split_ids'],norm)
    assert len(datasets['train'])==480 and budget(60)['supervised_fields']==28800
    sources=sorted(Path('stage6_control').glob('*.py'))+[Path('examples/train_stage6.py')]
    protocol={'config':config.to_dict(),'architecture':previous['architecture'],'normalization':norm.to_dict(),
        'split_ids':previous['split_ids'],'initial_state_sha256':state_hash(model),**parameter_counts(model),
        'optimizer':{'name':'Adam','lr':5e-4,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0},
        'objective':'one-step normalized MSE only','completed_budget':budget(60),
        'selection_rule':'Stage 5 selection_key: validation final AR mean L2, then mass error, then earlier epoch on exact ties',
        'interpretation':INTERPRETATION,'optional_e_ceiling_seconds':120,
        'training_archive_sha256':baseline['archive_sha256'],'stage5_protocol_sha256':sha256('runs/stage5/protocol.json'),
        'source_sha256':{str(p):sha256(p) for p in sources},'device':'cpu','platform':platform.platform(),
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)},
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    run.mkdir(parents=True)
    write_json(run/'protected_before.json',protected)
    write_json(run/'protocol.json',protocol)
    del model
    gate=tiny_gate(datasets['train'],norm,protocol['architecture'],config)
    write_json(run/'gate.json',gate)
    if not gate['passed']:
        write_json(run/'stopped.json',{'reason':'tiny_pair_gate_failed'})
        raise SystemExit('Sanity gate failed; no full training started')
    full_training(run,protocol,data,datasets['train'],norm,config)
    print('D selection frozen:',run/'selection.json',flush=True)


if __name__=='__main__':
    main()
