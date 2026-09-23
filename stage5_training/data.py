"""Windows indexed within frozen trajectory splits, never across them."""
import numpy as np
import torch
from torch.utils.data import Dataset


class RolloutWindowDataset(Dataset):
    def __init__(self,data,trajectory_ids,normalization,steps=5):
        ids=list(trajectory_ids)
        if not ids or len(set(ids))!=len(ids) or any(i<0 or i>=len(data['fields']) for i in ids):
            raise ValueError('Invalid trajectory IDs')
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<data['fields'].shape[1]:
            raise ValueError('Invalid rollout length')
        if not np.allclose(np.diff(data['times']),0.1,rtol=0,atol=1e-12):
            raise ValueError('Expected 0.1 time spacing')
        self.data=data;self.normalization=normalization;self.steps=steps
        self.windows=[(i,k) for i in ids for k in range(data['fields'].shape[1]-steps)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        i,k=self.windows[index];norm=self.normalization;n=self.data['fields'].shape[-1]
        start=self.data['fields'][i,k]
        field=torch.tensor(norm.field(start),dtype=torch.float32)[None]
        coeff=torch.tensor(norm.coefficients(self.data['coefficients'][i]),dtype=torch.float32)
        inputs=torch.cat((field,coeff[:,None,None].expand(3,n,n)),dim=0)
        target=torch.tensor(norm.field(self.data['fields'][i,k+1:k+1+self.steps]),dtype=torch.float32)[:,None]
        # Exact float64 start mass; it is a constant reference, not a detached prediction.
        initial_mass=torch.tensor(start.sum()*(2*np.pi/n)**2,dtype=torch.float64)
        return inputs,target,initial_mass,i,k
