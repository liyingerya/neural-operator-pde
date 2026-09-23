"""Training-only objectives and validation-only checkpoint selection."""
import hashlib
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from neural_operator.fno import FNO2d
from neural_operator.training import seed_everything
from stage4_evaluation.rollout import predict_trajectories
from stage4_evaluation.diagnostics import physical_diagnostics
from .objective import autoregressive_steps,losses


def fresh_model(config,architecture):
    seed_everything(config.seed,config.threads)
    return FNO2d(**architecture)


def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loader(dataset,config,shuffle=False):
    return DataLoader(dataset,batch_size=config.batch_size,shuffle=shuffle,num_workers=0,drop_last=False,
                      generator=torch.Generator().manual_seed(config.loader_seed))


def train_epoch(model,batches,optimizer,normalization,weight,config):
    model.train();totals={k:0. for k in ('total','field','mass')};count=0
    for inputs,target,m0,_,_ in batches:
        optimizer.zero_grad(set_to_none=True)
        predictions=autoregressive_steps(model,inputs,config.steps)
        value=losses(predictions,target,m0,normalization,weight,config.mass_epsilon)
        if not all(torch.isfinite(value[k]) for k in totals):
            raise FloatingPointError('Nonfinite rollout training loss')
        value['total'].backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Missing/nonfinite gradient')
        optimizer.step()
        for key in totals:
            totals[key]+=value[key].item()*len(inputs)
        count+=len(inputs)
    return {key:value/count for key,value in totals.items()}


@torch.no_grad()
def window_metrics(model,batches,normalization,weight,config):
    model.eval();totals={k:0. for k in ('total','field','mass')};count=0;relative=[];horizons=[]
    for inputs,target,m0,_,_ in batches:
        value=losses(autoregressive_steps(model,inputs,config.steps),target,m0,normalization,weight,config.mass_epsilon)
        for key in totals:
            totals[key]+=value[key].item()*len(inputs)
        prediction=normalization.inverse_field(value['predicted'].double())
        truth=normalization.inverse_field(target.double())
        relative.extend((torch.linalg.vector_norm((prediction-truth).flatten(2),dim=2)/torch.linalg.vector_norm(truth.flatten(2),dim=2).clamp_min(1e-12)).tolist())
        horizons.append(value['field_per_horizon'].tolist());count+=len(inputs)
    return {**{key:value/count for key,value in totals.items()},'physical_relative_l2':float(np.mean(relative)),
            'field_per_horizon':np.mean(horizons,axis=0).tolist()}


def gate_passed(initial,final,weight):
    finite=all(np.isfinite(final[k]) for k in ['field','mass','physical_relative_l2'])
    return bool(finite and final['field']<=0.05*initial['field'] and final['physical_relative_l2']<=0.05
                and (weight==0 or final['mass']<=0.1*initial['mass']))


def tiny_gate(dataset,normalization,architecture,weight,config):
    model=fresh_model(config,architecture)
    stride=dataset.data['fields'].shape[1]-config.steps
    indices=[i*stride for i in range(8)]
    batches=loader(Subset(dataset,indices),config)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    initial=window_metrics(model,batches,normalization,weight,config)
    started=time.perf_counter();history=[];final=initial
    for update in range(1,config.gate_updates+1):
        train_epoch(model,batches,optimizer,normalization,weight,config)
        if update%25==0 or update==config.gate_updates:
            final=window_metrics(model,batches,normalization,weight,config)
            history.append({'update':update,**final})
            print(f'gate lambda={weight:g} update={update} field={final["field"]:.6g} mass={final["mass"]:.6g} L2={final["physical_relative_l2"]:.6g}',flush=True)
            if gate_passed(initial,final,weight):
                break
    return {'passed':gate_passed(initial,final,weight),'lambda_mass':weight,'updates':update,
            'initial':initial,'final':final,'history':history,'windows':[list(dataset.windows[i]) for i in indices],
            'criteria':{'maximum_field_loss_ratio':0.05,'maximum_physical_relative_l2':0.05,
                        'maximum_mass_loss_ratio_for_C':0.1},'runtime_seconds':time.perf_counter()-started}


@torch.inference_mode()
def validation_metrics(model,data,validation_ids,normalization):
    fields=data['fields'][validation_ids]
    predictions,failures=predict_trajectories(model,normalization,fields,data['coefficients'][validation_ids],data['times'])
    if any(failures.values()):
        raise FloatingPointError('Nonfinite validation rollout')
    metrics=physical_diagnostics(predictions['autoregressive'],fields,fields[:,0])
    return {'final_relative_l2':float(metrics['relative_l2'][:,-1].mean()),
            'mean_relative_l2':float(metrics['relative_l2'][:,1:].mean(1).mean()),
            'final_mass_error':float(metrics['relative_mass_error'][:,-1].mean()),
            'relative_l2_by_horizon':metrics['relative_l2'].mean(0).tolist(),
            'mass_error_by_horizon':metrics['relative_mass_error'].mean(0).tolist()}


def selection_key(validation,epoch):
    # Field accuracy first; mass only breaks an exact field-error tie.
    return (validation['final_relative_l2'],validation['final_mass_error'],epoch)
