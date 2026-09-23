"""Evaluate frozen A/B/C/D with the unchanged Stage 4 numerical framework."""
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
from stage6_control.provenance import load_d,verify_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/stage6'))
    args=parser.parse_args();run=args.run
    d_model,d_norm,d_selection=load_d(run)
    protected=json.loads((run/'protected_before.json').read_text())
    verify_snapshot(protected)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    a_model,a_norm,baseline=load_frozen()
    stage5=json.loads(Path('docs/results/stage5_summary.json').read_text())
    datasets={'id_test':{'path':'data/dev/trajectories.npz','sha256':baseline['archive_sha256']}}
    for name in ('ood_manifest.json','resolution_manifest.json'):
        datasets.update(json.loads((Path('data/stage4')/name).read_text()))
    for value in datasets.values():verify_hash(value['path'],value['sha256'])
    output=run/'evaluation';output.mkdir(exist_ok=False)
    manifest={'selection_sha256':sha256(run/'selection.json'),'variants':{},
              'archive_sha256':{k:v['sha256'] for k,v in datasets.items()},
              'source_sha256':{str(p):sha256(p) for p in sorted(Path('stage6_control').glob('*.py'))+[Path(__file__).relative_to(Path.cwd())]}}
    for variant in ('A','B','C','D'):
        if variant=='A':model,norm,chosen=a_model,a_norm,baseline['selected_checkpoint']
        elif variant=='D':model,norm,chosen=d_model,d_norm,d_selection
        else:model,norm,chosen=load_candidate(Path('runs/stage5'),variant)
        folder=output/variant;folder.mkdir();results={};resolution={};start=time.perf_counter()
        for name,value in datasets.items():
            data=load_dataset(value['path'])
            ids=baseline['split_ids']['test'] if name=='id_test' else None
            result,predictions=evaluate_archive(model,norm,data,name,ids)
            result.update(variant=variant,checkpoint_sha256=chosen['sha256'],archive_sha256=value['sha256'])
            write_json(folder/f'{name}.json',result)
            np.savez_compressed(folder/f'{name}_predictions.npz',**predictions)
            results[name]=result
            if name.startswith('resolution_'):resolution[name]=(data,predictions)
            print(variant,name,'AR',result['summary']['autoregressive']['relative_l2']['mean'][-1],flush=True)
        results['high_velocity']=combine_results([results[f'velocity_{q}'] for q in ('pp','pn','np','nn')],'high_velocity')
        write_json(folder/'high_velocity.json',results['high_velocity'])
        coarse,pc=resolution['resolution_64'];fine,pf=resolution['resolution_128'];verify_matched(coarse,fine)
        comparison={}
        for method in ('teacher_forced','autoregressive'):
            delta=np.asarray(results['resolution_128']['metrics'][method]['relative_l2'])-np.asarray(results['resolution_64']['metrics'][method]['relative_l2'])
            shared=np.sqrt(((pc[method]-pf[method][:,:,::2,::2])**2).sum(axis=(-2,-1)))/np.maximum(np.sqrt((pf[method][:,:,::2,::2]**2).sum(axis=(-2,-1))),1e-12)
            comparison[method]={'native_error_difference_128_minus_64':delta.mean(0).tolist(),
                                'prediction_shared_grid_relative_l2':shared.mean(0).tolist()}
        write_json(folder/'resolution_comparison.json',comparison)
        if variant!='D':
            for name,result in results.items():
                previous=stage5['models'][variant]['regimes'][name]
                for metric,values in result['summary']['autoregressive'].items():
                    np.testing.assert_allclose(values['mean'],previous['horizon_means']['autoregressive'][metric],rtol=1e-6,atol=1e-9)
                np.testing.assert_allclose(result['summary']['teacher_forced']['relative_l2']['mean'],previous['horizon_means']['teacher_forced']['relative_l2'],rtol=1e-6,atol=1e-9)
        manifest['variants'][variant]={'checkpoint_sha256':chosen['sha256'],'runtime_seconds':time.perf_counter()-start}
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
