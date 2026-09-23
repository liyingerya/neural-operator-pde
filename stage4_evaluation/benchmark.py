"""CPU timing of matched problems; numerical precision and batching are explicit."""
import os
import platform
import subprocess
import time
import numpy as np
import torch
from advection_diffusion import solve
from .rollout import prepare_input


def device_context():
    def sysctl(key):
        try:
            return subprocess.check_output(['sysctl','-n',key],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            return None
    return {'device':'cpu','platform':platform.platform(),'cpu_model':sysctl('machdep.cpu.brand_string'),
            'physical_cores':sysctl('hw.physicalcpu'),'logical_cores':os.cpu_count(),
            'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)},
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            'torch_build':torch.__config__.show()}


def measure(function,repeats=30,warmups=5,min_group_seconds=0.02):
    if repeats<1 or warmups<0 or min_group_seconds<=0:
        raise ValueError('Invalid timing configuration')
    for _ in range(warmups):
        function()
    start=time.perf_counter(); function(); pilot=time.perf_counter()-start
    iterations=max(1,min(1000,int(np.ceil(min_group_seconds/max(pilot,1e-9)))))
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter()-start)/iterations)
    return {'median_seconds':float(np.median(samples)),'q25_seconds':float(np.quantile(samples,0.25)),
            'q75_seconds':float(np.quantile(samples,0.75)),'samples_seconds':samples,
            'iterations_per_group':iterations,'repeats':repeats,'warmups':warmups}


@torch.inference_mode()
def benchmark_dataset(model,normalization,data,ids):
    model.eval()
    output={}
    for batch in (1,8):
        chosen=list(ids)[:batch]
        fields=data['fields'][chosen,0]; coefficients=data['coefficients'][chosen]
        dt=data['base_dt'][chosen]
        prepared=prepare_input(fields,coefficients,normalization)
        def pde():
            return [solve(u,0.1,*c,float(step)) for u,c,step in zip(fields,coefficients,dt)]
        def forward():
            return model(prepared)
        def end_to_end():
            prediction=model(prepare_input(fields,coefficients,normalization))
            return normalization.inverse_field(prediction[:,0].double()).numpy()
        timings={name:measure(fn) for name,fn in [('solver',pde),('fno_forward',forward),('fno_end_to_end',end_to_end)]}
        for value in timings.values():
            value['problems_per_second']=len(chosen)/value['median_seconds']
        timings['speedup_end_to_end']=timings['solver']['median_seconds']/timings['fno_end_to_end']['median_seconds']
        timings['speedup_forward']=timings['solver']['median_seconds']/timings['fno_forward']['median_seconds']
        timings['trajectory_ids']=chosen
        timings['base_dt']=dt.tolist()
        output[str(batch)]=timings
    return output
