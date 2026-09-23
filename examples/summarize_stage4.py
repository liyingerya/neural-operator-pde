"""Create a compact public report from saved Stage 4 evaluation artifacts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from stage4_evaluation.checkpoint import sha256,write_json

NAMES=['id_test','fresh_id','high_nu','high_velocity','velocity_pp','velocity_pn','velocity_np','velocity_nn','resolution_64','resolution_128']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/stage4'))
    parser.add_argument('--output',type=Path,default=Path('docs/results/stage4_summary.json'))
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run=args.run
    results={name:json.loads((run/f'{name}.json').read_text()) for name in NAMES}
    benchmark=json.loads((run/'benchmark.json').read_text())
    hardware=json.loads((run/'hardware.json').read_text()) if (run/'hardware.json').exists() else {}
    device={k:v for k,v in benchmark['device'].items() if k!='torch_build'}
    device.update({k:hardware[k] for k in ('cpu_model','physical_cores','logical_cores') if k in hardware})
    manifests={}
    for filename in ('ood_manifest.json','resolution_manifest.json'):
        manifests.update(json.loads((Path('data/stage4')/filename).read_text()))
    runtime={phase:json.loads((run/f'{phase}_context.json').read_text())['runtime_seconds']
             for phase in ('id','ood','resolution')}
    runtime['benchmark']=benchmark['runtime_seconds']
    runtime['generation']=sum(v['runtime_seconds'] for v in manifests.values())
    summary={'schema_version':1,'checkpoint_sha256':benchmark['checkpoint_sha256'],'checkpoint_epoch':31,
        'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'normalization':json.loads((run/'id_context.json').read_text())['normalization'],
        'normalization_refitted':False,'device':device,'recorded_phase_runtime_seconds':runtime,
        'recorded_phase_runtime_sum_seconds':sum(runtime.values()),
        'runtime_note':'Sum of timed generation, evaluation and benchmark phases; excludes setup, tests, plotting, loading outside timers, and development.',
        'datasets':{k:{key:value for key,value in v.items() if key!='path'} for k,v in manifests.items()},
        'regimes':{},'benchmark':{},
        'source_sha256':{str(p):sha256(p) for p in sorted(Path('stage4_evaluation').glob('*.py'))+sorted(Path('examples').glob('*stage4*.py'))},
        'artifact_sha256':{str(p.relative_to(run)):sha256(p) for p in sorted(run.glob('*.json'))}}
    for name,r in results.items():
        metrics=r['metrics'];ar=np.asarray(metrics['autoregressive']['relative_l2'],dtype=float)
        finite=np.isfinite(ar[:,-1]);worst=int(np.argmax(np.where(finite,ar[:,-1],-np.inf))) if finite.any() else None
        tf=np.asarray(metrics['teacher_forced']['relative_l2'],dtype=float)[:,1:]
        means={method:{metric:stats['mean'] for metric,stats in r['summary'][method].items()}
               for method in ('teacher_forced','autoregressive','persistence','reference')}
        summary['regimes'][name]={'trajectory_count':len(r['trajectory_ids']),'times':r['times'],
            'teacher_forced_mean_over_horizons':float(tf.mean()) if np.isfinite(tf).all() else None,
            'horizon_means':means,'feedback_ratio_of_means':r['feedback_amplification']['ratio_of_mean_errors'],
            'feedback_mean_of_trajectory_ratios':r['feedback_amplification']['summary']['mean'],
            'autoregressive_horizon_spread':{metric:r['summary']['autoregressive'][metric] for metric in ['relative_l2','relative_mass_error','minimum','maximum']},
            'worst_final_trajectory':None if worst is None else {'id':r['trajectory_ids'][worst],'relative_l2':float(ar[worst,-1])},
            'failures':r['failures']}
    for name,workload in benchmark['workloads'].items():
        summary['benchmark'][name]={}
        for batch,timing in workload.items():
            summary['benchmark'][name][batch]={key:({k:v for k,v in value.items() if k!='samples_seconds'} if isinstance(value,dict) else value)
                                                for key,value in timing.items()}
    comparison=json.loads((run/'resolution_comparison.json').read_text())
    summary['resolution_comparison']={key:value for key,value in comparison.items() if key not in ('reference_shared_grid_relative_l2','prediction_shared_grid_relative_l2','native_error_difference_128_minus_64')}
    summary['resolution_comparison']['paired_final_native_error_difference_128_minus_64']={
        method:{'mean':float(np.mean(np.asarray(values)[:,-1])),'min':float(np.min(np.asarray(values)[:,-1])),
                'max':float(np.max(np.asarray(values)[:,-1]))}
        for method,values in comparison['native_error_difference_128_minus_64'].items()}
    summary['resolution_comparison']['prediction_shared_grid_mean']={method:np.asarray(values).mean(0).tolist()
        for method,values in comparison['prediction_shared_grid_relative_l2'].items()}
    write_json(args.output,summary)
    csv_path=args.output.with_name('stage4_final_by_trajectory.csv')
    columns=['regime','trajectory_id','teacher_forced_final_l2','autoregressive_final_l2','persistence_final_l2','feedback_ratio',
             'ar_relative_mass_error','ar_signed_mass_drift','ar_accumulated_absolute_mass_change','ar_negative_fraction',
             'ar_negative_mass_fraction','ar_minimum','ar_maximum']
    with csv_path.open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        for name,r in results.items():
            if name.startswith('velocity_'):
                continue  # Combined high_velocity IDs already retain the quadrant.
            for i,identifier in enumerate(r['trajectory_ids']):
                values=[name,identifier]+[r['metrics'][method]['relative_l2'][i][-1] for method in ['teacher_forced','autoregressive','persistence']]
                values+=[r['feedback_amplification']['per_trajectory'][i][-1]]
                values+=[r['metrics']['autoregressive'][metric][i][-1] for metric in ['relative_mass_error','signed_mass_drift','accumulated_absolute_mass_change','negative_fraction','negative_mass_fraction','minimum','maximum']]
                writer.writerow(values)
    print(args.output);print(csv_path)


if __name__=='__main__':
    main()
