"""Predeclared rollout-objective ablations; no test/OOD evaluation here."""
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
from stage4_evaluation.checkpoint import load_frozen,sha256,write_json,APPROVED_HASH
from stage5_training.config import TrainingConfig,LAMBDAS
from stage5_training.data import RolloutWindowDataset
from stage5_training.training import (fresh_model,state_hash,loader,tiny_gate,train_epoch,
                                       validation_metrics,selection_key)


def save_progress(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def run_training(output,config,baseline,normalization,architecture,data,dataset,weight,initial_hash):
    output.mkdir()
    model=fresh_model(config,architecture)
    assert state_hash(model)==initial_hash
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    batches=loader(dataset,config,shuffle=True)
    history=[];best=None;selection=None;start=time.perf_counter()
    manifest={'status':'training','config':config.to_dict(),'architecture':architecture,
              **parameter_counts(model),'lambda_mass':weight,'initial_state_sha256':initial_hash,
              'normalization':normalization.to_dict(),'split_ids':baseline['split_ids'],
              'historical_checkpoint_sha256':APPROVED_HASH,'training_archive_sha256':baseline['archive_sha256'],
              'checkpoint_selection':'minimum validation final t=1 autoregressive relative L2; mass then earliest epoch break exact ties',
              'selection':None}
    write_json(output/'manifest.json',manifest)
    for epoch in range(1,config.epochs+1):
        tick=time.perf_counter()
        training=train_epoch(model,batches,optimizer,normalization,weight,config)
        validation=validation_metrics(model,data,baseline['split_ids']['validation'],normalization)
        history.append({'epoch':epoch,'training':training,'validation':validation,'runtime_seconds':time.perf_counter()-tick})
        key=selection_key(validation,epoch)
        if best is None or key<best:
            best=key
            torch.save({'model_state_dict':model.state_dict(),'architecture':architecture,
                'normalization':normalization.to_dict(),'split_ids':baseline['split_ids'],
                'config':config.to_dict(),'lambda_mass':weight,'epoch':epoch,
                'validation':validation,'initial_state_sha256':initial_hash},output/'best.pt')
            selection={'epoch':epoch,'validation':validation,'checkpoint':'best.pt','sha256':sha256(output/'best.pt')}
        save_progress(output/'history.json',history)
        manifest['selection']=selection;save_progress(output/'manifest.json',manifest)
        print(f'{output.name} epoch={epoch}/{config.epochs} field={training["field"]:.6g} mass={training["mass"]:.6g} val_AR={validation["final_relative_l2"]:.6g} val_mass={validation["final_mass_error"]:.6g} seconds={history[-1]["runtime_seconds"]:.2f}',flush=True)
    manifest.update(status='complete',runtime_seconds=time.perf_counter()-start)
    save_progress(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/stage5'))
    args=parser.parse_args();config=TrainingConfig()
    historical,normalization,baseline=load_frozen();architecture=historical.config
    del historical
    args.output.mkdir(parents=True,exist_ok=False)
    data=load_dataset('data/dev/trajectories.npz')
    dataset=RolloutWindowDataset(data,baseline['split_ids']['train'],normalization,config.steps)
    model=fresh_model(config,architecture);initial_hash=state_hash(model)
    paths=sorted(Path('stage5_training').glob('*.py'))+[Path(__file__)]
    protocol={'config':config.to_dict(),'lambda_candidates':list(LAMBDAS),'optimizer':'Adam',
        'weight_decay':0,'scheduler':None,'gradient_clipping':False,'initialization':'Stage 3 deterministic initialization procedure, same seed 2028 and PyTorch version; never pretrained weights',
        'initial_state_sha256':initial_hash,'architecture':architecture,**parameter_counts(model),
        'training_windows':len(dataset),'validation_windows':len(baseline['split_ids']['validation'])*(11-config.steps),
        'split_ids':baseline['split_ids'],'normalization':normalization.to_dict(),
        'historical_checkpoint_sha256':APPROVED_HASH,'training_archive_sha256':baseline['archive_sha256'],
        'device':'cpu','platform':platform.platform(),'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)},
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p.relative_to(Path.cwd()) if p.is_absolute() else p):sha256(p) for p in paths},
        'selection_rule':'Across epochs and C lambdas: lowest validation final-t=1 AR relative L2, then mass error, then earliest epoch, then smaller lambda on exact ties.',
        'evaluation_firewall':'No Stage 4 results or OOD archive are read by this training script.'}
    write_json(args.output/'protocol.json',protocol);del model
    gates=[]
    for weight in (0.0,*LAMBDAS):
        gate=tiny_gate(dataset,normalization,architecture,weight,config)
        gates.append(gate);write_json(args.output/f'gate_{weight:g}.json',gate)
        if not gate['passed']:
            write_json(args.output/'stopped.json',{'reason':'tiny_overfit_gate_failed','lambda_mass':weight})
            raise SystemExit('Tiny-window gate failed; no full training started.')
    manifests=[]
    for weight in (0.0,*LAMBDAS):
        name='rollout' if weight==0 else f'mass_{weight:g}'
        manifests.append(run_training(args.output/name,config,baseline,normalization,architecture,data,dataset,weight,initial_hash))
    best=min(manifests[1:],key=lambda m:(*selection_key(m['selection']['validation'],m['selection']['epoch']),m['lambda_mass']))
    write_json(args.output/'selection.json',{'B':{'directory':'rollout',**manifests[0]['selection']},
        'C':{'directory':f'mass_{best["lambda_mass"]:g}','lambda_mass':best['lambda_mass'],**best['selection']},
        'validation_comparison':[{'lambda_mass':m['lambda_mass'],**m['selection'],'runtime_seconds':m['runtime_seconds']} for m in manifests],
        'all_gates_passed':True,'test_or_ood_used':False})
    print('Selection frozen:',args.output/'selection.json',flush=True)


if __name__=='__main__':
    main()
