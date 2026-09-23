"""One-step MSE only; Stage 5 rollout validation selects checkpoints."""
import time
import numpy as np
import torch
from torch.utils.data import Subset
from neural_operator.training import train_epoch as one_step_epoch, evaluate, overfit_passed
from stage5_training.training import fresh_model, loader
from stage4_evaluation.rollout import predict_trajectories
from stage4_evaluation.diagnostics import physical_diagnostics


def train_epoch(model,batches,optimizer):
    value=one_step_epoch(model,batches,optimizer)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Missing/nonfinite one-step gradient')
    return value


def tiny_gate(dataset,norm,architecture,config):
    model=fresh_model(config,architecture)
    indices=[i*10 for i in range(8)]
    batches=loader(Subset(dataset,indices),config)
    initial=evaluate(model,batches,norm)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    start=time.perf_counter();history=[];final=initial
    for update in range(1,config.gate_updates+1):
        train_epoch(model,batches,optimizer)
        if update%25==0 or update==config.gate_updates:
            final=evaluate(model,batches,norm)
            history.append({'update':update,'normalized_mse':final['normalized_mse'],'relative_l2':final['relative_l2']})
            print(f'gate update={update} MSE={final["normalized_mse"]:.6g} physical_L2={final["relative_l2"]:.6g}',flush=True)
            if overfit_passed(initial,final):
                break
    return {'passed':overfit_passed(initial,final),'updates':update,'initial':initial,'final':final,
            'pairs':[list(dataset.pairs[i]) for i in indices],'history':history,
            'criteria':{'maximum_mse_ratio':0.05,'maximum_physical_relative_l2':0.05},
            'runtime_seconds':time.perf_counter()-start}


@torch.inference_mode()
def validation_metrics(model,data,ids,norm):
    fields=data['fields'][ids]
    outputs,failures=predict_trajectories(model,norm,fields,data['coefficients'][ids],data['times'])
    if any(failures.values()):
        raise FloatingPointError('Nonfinite validation prediction')
    ar=physical_diagnostics(outputs['autoregressive'],fields,fields[:,0])
    tf=physical_diagnostics(outputs['teacher_forced'],fields,fields[:,0])
    return {'final_relative_l2':float(ar['relative_l2'][:,-1].mean()),
            'final_mass_error':float(ar['relative_mass_error'][:,-1].mean()),
            'one_step_relative_l2':float(tf['relative_l2'][:,1:].mean(1).mean()),
            'relative_l2_by_horizon':ar['relative_l2'].mean(0).tolist(),
            'mass_error_by_horizon':ar['relative_mass_error'].mean(0).tolist()}
