"""Evaluate the frozen Stage 3 checkpoint; no fitting or corrections."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from dataset_generation import load_dataset
from stage4_evaluation.checkpoint import load_frozen,write_json,sha256,verify_hash
from stage4_evaluation.rollout import evaluate_archive,combine_results
from stage4_evaluation.ood import verify_matched
from stage4_evaluation.benchmark import device_context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['id','ood','resolution'],required=True)
    parser.add_argument('--output',type=Path,default=Path('runs/stage4'))
    parser.add_argument('--data',type=Path,default=Path('data/stage4'))
    args=parser.parse_args()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    model,norm,baseline=load_frozen()
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    datasets={}
    if args.phase=='id':
        datasets['id_test']=(Path('data/dev/trajectories.npz'),baseline['split_ids']['test'])
    else:
        name='ood_manifest.json' if args.phase=='ood' else 'resolution_manifest.json'
        manifest=json.loads((args.data/name).read_text())
        for key,value in manifest.items():
            path=Path(value['path']); verify_hash(path,value['sha256'])
            datasets[key]=(path,None)
    results={}; loaded={}; predictions={}
    for name,(path,ids) in datasets.items():
        result_path=args.output/f'{name}.json'
        if result_path.exists():
            raise FileExistsError(result_path)
        data=load_dataset(path)
        result,outputs=evaluate_archive(model,norm,data,name,ids)
        result.update(archive_sha256=sha256(path),checkpoint_sha256=baseline['selected_checkpoint']['sha256'])
        write_json(result_path,result)
        np.savez_compressed(args.output/f'{name}_predictions.npz',**outputs)
        results[name]=result
        if args.phase=='resolution':
            loaded[name]=data; predictions[name]=outputs
        print(name,{method:result['summary'][method]['relative_l2']['mean'][-1]
                    for method in ('teacher_forced','autoregressive','persistence')},flush=True)
    if args.phase=='ood':
        combined=combine_results([results[f'velocity_{q}'] for q in ('pp','pn','np','nn')],'high_velocity')
        write_json(args.output/'high_velocity.json',combined)
    if args.phase=='resolution':
        coarse,fine=loaded['resolution_64'],loaded['resolution_128']
        verify_matched(coarse,fine)
        fine_shared=fine['fields'][:,:,::2,::2]
        def relative(a,b):
            return np.sqrt(((a-b)**2).sum(axis=(-2,-1)))/np.maximum(np.sqrt((b*b).sum(axis=(-2,-1))),1e-12)
        discrepancy=relative(coarse['fields'],fine_shared)
        comparison={'matched_parameters_and_initial_fields':True,
                    'reference_shared_grid_relative_l2':discrepancy.tolist(),
                    'reference_shared_grid_mean':discrepancy.mean(0).tolist(),
                    'native_error_difference_128_minus_64':{},'prediction_shared_grid_relative_l2':{}}
        for method in ('teacher_forced','autoregressive'):
            coarse_error=np.asarray(results['resolution_64']['metrics'][method]['relative_l2'])
            fine_error=np.asarray(results['resolution_128']['metrics'][method]['relative_l2'])
            comparison['native_error_difference_128_minus_64'][method]=(fine_error-coarse_error).tolist()
            comparison['prediction_shared_grid_relative_l2'][method]=relative(
                predictions['resolution_64'][method],predictions['resolution_128'][method][:,:,::2,::2]).tolist()
        write_json(args.output/'resolution_comparison.json',comparison)
    write_json(args.output/f'{args.phase}_context.json',{'runtime_seconds':time.perf_counter()-start,
        'device':device_context(),'checkpoint_sha256':baseline['selected_checkpoint']['sha256'],
        'normalization':norm.to_dict(),'normalization_refitted':False})


if __name__=='__main__':
    main()
