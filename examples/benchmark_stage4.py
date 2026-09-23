"""Steady-state CPU benchmarks; loading and diagnostics are excluded."""
import argparse
import json
from pathlib import Path
import time
import torch
from dataset_generation import load_dataset
from stage4_evaluation.checkpoint import load_frozen,write_json,verify_hash
from stage4_evaluation.benchmark import benchmark_dataset,device_context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/stage4/benchmark.json'))
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model,norm,baseline=load_frozen()
    start=time.perf_counter()
    workloads={'id_test':('data/dev/trajectories.npz',baseline['split_ids']['test'])}
    manifest=json.loads(Path('data/stage4/ood_manifest.json').read_text())
    for name,value in manifest.items():
        verify_hash(value['path'],value['sha256'])
        workloads[name]=(value['path'],list(range(value['config']['num_trajectories'])))
    result={'device':device_context(),'checkpoint_sha256':baseline['selected_checkpoint']['sha256'],
            'interval':0.1,'solver_dtype':'float64','fno_dtype':'float32/complex64',
            'solver_batching':'sequential loop','workloads':{}}
    for name,(path,ids) in workloads.items():
        result['workloads'][name]=benchmark_dataset(model,norm,load_dataset(path),ids)
        print(name,{batch:timing['speedup_end_to_end'] for batch,timing in result['workloads'][name].items()},flush=True)
    result['runtime_seconds']=time.perf_counter()-start
    write_json(args.output,result)


if __name__=='__main__':
    main()
