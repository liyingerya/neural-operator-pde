"""Evaluation-only datasets through the unchanged Stage 2 public API."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from dataset_generation import GenerationConfig, generate_dataset, save_dataset
from advection_diffusion import solve
from .checkpoint import sha256, write_json


def configurations(resolution=False):
    if resolution:
        return {f'resolution_{n}':GenerationConfig(num_trajectories=32,grid_size=n,seed=4120)
                for n in (64,128)}
    configs = {'fresh_id':GenerationConfig(seed=4100),
               'high_nu':GenerationConfig(seed=4101,nu_range=(0.055,0.08))}
    for offset,(label,sx,sy) in enumerate([('pp',1,1),('pn',1,-1),('np',-1,1),('nn',-1,-1)]):
        bounds = lambda sign:(1.05,1.4) if sign>0 else (-1.4,-1.05)
        configs[f'velocity_{label}'] = GenerationConfig(num_trajectories=16,seed=4110+offset,
                                                       cx_range=bounds(sx),cy_range=bounds(sy))
    return configs


def internal_steps(duration,dt):
    t=0.0; count=0
    while t<duration:
        step=min(dt,duration-t)
        if t+step==t:
            raise ValueError('Timestep cannot advance the clock')
        t=duration if step==duration-t else t+step
        count+=1
    return count


def numerical_checks(data):
    fields=data['fields']; n=int(data['grid_size'])
    mass=fields.sum(axis=(-2,-1))*(2*np.pi/n)**2
    refinement=[]
    duration=float(data['times'][1]-data['times'][0])
    for i in range(min(4,len(fields))):
        refined=solve(fields[i,0],duration,*data['coefficients'][i],data['base_dt'][i]/2)
        relative=np.linalg.norm(refined-fields[i,1])/max(np.linalg.norm(refined),1e-12)
        refinement.append({'trajectory_id':i,'relative_l2_base_vs_half_dt':float(relative)})
    return {'finite':bool(np.isfinite(fields).all()),'minimum':float(fields.min()),'maximum':float(fields.max()),
            'negative_entries':int(np.count_nonzero(fields<0)),
            'max_absolute_mass_drift':float(np.max(np.abs(mass-mass[:,:1]))),
            'base_dt_range':[float(data['base_dt'].min()),float(data['base_dt'].max())],
            'first_interval_step_counts':[internal_steps(duration,float(dt)) for dt in data['base_dt']],
            'timestep_refinement':refinement}


def verify_matched(coarse,fine):
    for name in ('coefficients','blob_counts','blob_mask','blob_centers','blob_widths','blob_amplitudes','times'):
        if not np.array_equal(coarse[name],fine[name]):
            raise ValueError(f'Resolution pair mismatch: {name}')
    if not np.array_equal(coarse['fields'][:,0],fine['fields'][:,0,::2,::2]):
        raise ValueError('Initial conditions do not match at shared grid points')


def generate_evaluation_data(directory, resolution=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    manifest={}; generated={}
    for name,config in configurations(resolution).items():
        path=directory/f'{name}.npz'
        if path.exists():
            raise FileExistsError(path)
        start=time.perf_counter()
        data=generate_dataset(config)
        checks=numerical_checks(data)
        save_dataset(path,data)
        manifest[name]={'path':str(path),'sha256':sha256(path),'config':asdict(config),
                        'evaluation_only':True,'numerical_checks':checks,
                        'runtime_seconds':time.perf_counter()-start}
        if resolution:
            generated[name]=data
        print(f'Generated {name}: {config.num_trajectories} trajectories, {time.perf_counter()-start:.2f}s',flush=True)
    if resolution:
        verify_matched(generated['resolution_64'],generated['resolution_128'])
    write_json(directory/('resolution_manifest.json' if resolution else 'ood_manifest.json'),manifest)
    return manifest
