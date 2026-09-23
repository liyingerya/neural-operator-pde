"""Re-evaluate A/B/C only after validation-only selection has been frozen."""
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
from stage5_training.checkpoint import load_candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/stage5'))
    args=parser.parse_args();run=args.run
    selection=json.loads((run/'selection.json').read_text())
    if selection['test_or_ood_used'] or not selection['all_gates_passed']:
        raise ValueError('Training selection must be frozen before evaluation')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    baseline_model,norm,baseline=load_frozen()
    stage4=json.loads(Path('docs/results/stage4_summary.json').read_text())
    for path,digest in stage4['artifact_sha256'].items():
        verify_hash(Path('runs/stage4')/path,digest)
    datasets={'id_test':{'path':'data/dev/trajectories.npz','sha256':baseline['archive_sha256']}}
    for name in ('ood_manifest.json','resolution_manifest.json'):
        datasets.update(json.loads((Path('data/stage4')/name).read_text()))
    for value in datasets.values():verify_hash(value['path'],value['sha256'])
    output=run/'evaluation';output.mkdir(exist_ok=False)
    evaluation_manifest={'selection_sha256':sha256(run/'selection.json'),
                         'archive_sha256':{name:value['sha256'] for name,value in datasets.items()},
                         'source_sha256':{str(p):sha256(p) for p in sorted(Path('stage5_training').glob('*.py'))+[Path('examples/evaluate_stage5.py')]},
                         'variants':{}}
    for variant in ('A','B','C'):
        model,normalization,chosen=(baseline_model,norm,baseline['selected_checkpoint']) if variant=='A' else load_candidate(run,variant)
        folder=output/variant;folder.mkdir();start=time.perf_counter();results={};resolution={}
        for name,value in datasets.items():
            data=load_dataset(value['path'])
            ids=baseline['split_ids']['test'] if name=='id_test' else None
            result,predictions=evaluate_archive(model,normalization,data,name,ids)
            result.update(variant=variant,checkpoint_sha256=chosen['sha256'],archive_sha256=value['sha256'])
            write_json(folder/f'{name}.json',result)
            np.savez_compressed(folder/f'{name}_predictions.npz',**predictions)
            results[name]=result
            if name.startswith('resolution_'):
                resolution[name]=(data,predictions)
            print(variant,name,'TF mean',np.mean(result['summary']['teacher_forced']['relative_l2']['mean'][1:]),
                  'AR final',result['summary']['autoregressive']['relative_l2']['mean'][-1],flush=True)
        combined=combine_results([results[f'velocity_{q}'] for q in ('pp','pn','np','nn')],'high_velocity')
        write_json(folder/'high_velocity.json',combined)
        coarse,pc=resolution['resolution_64'];fine,pf=resolution['resolution_128']
        verify_matched(coarse,fine)
        comparison={}
        for method in ('teacher_forced','autoregressive'):
            differences=np.asarray(results['resolution_128']['metrics'][method]['relative_l2'])-np.asarray(results['resolution_64']['metrics'][method]['relative_l2'])
            relative=np.sqrt(((pc[method]-pf[method][:,:,::2,::2])**2).sum(axis=(-2,-1)))/np.maximum(np.sqrt((pf[method][:,:,::2,::2]**2).sum(axis=(-2,-1))),1e-12)
            comparison[method]={'native_error_difference_128_minus_64':differences.tolist(),
                                'prediction_shared_grid_relative_l2':relative.tolist()}
        write_json(folder/'resolution_comparison.json',comparison)
        if variant=='A':
            for name,result in results.items():
                np.testing.assert_allclose(result['summary']['autoregressive']['relative_l2']['mean'],
                    stage4['regimes'][name]['horizon_means']['autoregressive']['relative_l2'],rtol=1e-6,atol=1e-9)
        evaluation_manifest['variants'][variant]={'checkpoint_sha256':chosen['sha256'],
                                                  'runtime_seconds':time.perf_counter()-start}
    write_json(output/'manifest.json',evaluation_manifest)


if __name__=='__main__':
    main()
