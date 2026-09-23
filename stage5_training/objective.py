"""Full backpropagation through autoregressive feedback and physical mass."""
import torch
from .config import validate_lambda


def autoregressive_steps(model,inputs,steps):
    condition=inputs[:,1:]
    state=inputs[:,:1]
    predictions=[]
    for _ in range(steps):
        state=model(torch.cat((state,condition),dim=1))
        predictions.append(state)
    return predictions


def losses(predictions,target,initial_mass,normalization,lambda_mass,epsilon=1e-12):
    weight=validate_lambda(lambda_mass)
    predicted=torch.stack(predictions,dim=1)
    per_horizon=(predicted-target).square().mean(dim=(0,2,3,4))
    field=per_horizon.mean()
    physical=normalization.inverse_field(predicted.double())
    area=(2*torch.pi/predicted.shape[-1])**2
    mass=physical.sum(dim=(-3,-2,-1))*area
    mass_loss=((mass-initial_mass[:,None]).square()/initial_mass[:,None].square().clamp_min(epsilon)).mean()
    total=field+weight*mass_loss
    return {'total':total,'field':field,'mass':mass_loss,'field_per_horizon':per_horizon,'predicted':predicted}
