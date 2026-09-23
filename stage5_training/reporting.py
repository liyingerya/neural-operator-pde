"""Public summaries and figures from already-selected, already-evaluated models."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from stage4_evaluation.checkpoint import sha256,write_json

REGIMES=['id_test','fresh_id','high_nu','high_velocity','velocity_pp','velocity_pn','velocity_np','velocity_nn','resolution_64','resolution_128']
LABELS={'A':'Historical one-step','B':'Rollout trained','C':'Rollout + mass'}
COLORS={'A':'#64748b','B':'#2563eb','C':'#d97706'}


def report(run=Path('runs/stage5'),public=Path('docs/results/stage5_summary.json')):
    run=Path(run);public=Path(public)
    protocol=json.loads((run/'protocol.json').read_text());selection=json.loads((run/'selection.json').read_text())
    evaluation=json.loads((run/'evaluation/manifest.json').read_text())
    if evaluation['selection_sha256']!=sha256(run/'selection.json'):
        raise ValueError('Selection changed after evaluation')
    results={variant:{name:json.loads((run/f'evaluation/{variant}/{name}.json').read_text()) for name in REGIMES} for variant in LABELS}
    gates=[{k:v for k,v in json.loads((run/f'gate_{weight:g}.json').read_text()).items() if k!='history'} for weight in [0,0.01,0.1,1.]]
    output={'protocol':protocol,'selection':selection,'gates':gates,'evaluation_manifest':evaluation,'models':{},
            'hardware_context':{k:v for k,v in json.loads(Path('runs/stage4/hardware.json').read_text()).items() if k in ('cpu_model','physical_cores','logical_cores')},
            'source_sha256':{str(p):sha256(p) for p in sorted(Path('stage5_training').glob('*.py'))+sorted(Path('examples').glob('*stage5*.py'))}}
    for variant,regimes in results.items():
        resolution=json.loads((run/f'evaluation/{variant}/resolution_comparison.json').read_text())
        output['models'][variant]={'label':LABELS[variant],'regimes':{},
            'resolution_comparison':{method:{metric:np.asarray(values).mean(axis=0).tolist()
                for metric,values in metrics.items()} for method,metrics in resolution.items()}}
        for name,r in regimes.items():
            output['models'][variant]['regimes'][name]={'trajectory_count':len(r['trajectory_ids']),'times':r['times'],
                'one_step_mean_relative_l2':float(np.asarray(r['metrics']['teacher_forced']['relative_l2'])[:,1:].mean(1).mean()),
                'horizon_means':{method:{metric:values[metric]['mean'] for metric in (values if method=='autoregressive' else ('relative_l2','relative_mass_error'))} for method,values in r['summary'].items()},
                'feedback_ratio_of_means':r['feedback_amplification']['ratio_of_mean_errors'],
                'minimum_across_trajectories':r['summary']['autoregressive']['minimum']['min'],
                'maximum_across_trajectories':r['summary']['autoregressive']['maximum']['max'],
                'worst_final_trajectory':{'id':r['trajectory_ids'][int(np.argmax(np.asarray(r['metrics']['autoregressive']['relative_l2'])[:,-1]))],
                    'relative_l2':float(np.max(np.asarray(r['metrics']['autoregressive']['relative_l2'])[:,-1]))},
                'failures':r['failures']}
    write_json(public,output)
    columns=['model','one_step_id_l2','final_id_ar_l2','final_id_mass_error','high_velocity_final_ar_l2','resolution_64_final_ar_l2','resolution_128_final_ar_l2']
    with public.with_name('stage5_ablation.csv').open('x',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(columns)
        for variant in LABELS:
            r=output['models'][variant]['regimes'];id_result=r['id_test']
            writer.writerow([variant,id_result['one_step_mean_relative_l2'],id_result['horizon_means']['autoregressive']['relative_l2'][-1],
                id_result['horizon_means']['autoregressive']['relative_mass_error'][-1],
                *[r[name]['horizon_means']['autoregressive']['relative_l2'][-1] for name in ['high_velocity','resolution_64','resolution_128']]])
    os.environ.setdefault('MPLCONFIGDIR',str((run/'mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    figures=run/'figures';figures.mkdir(exist_ok=True)
    def save(fig,name):fig.savefig(figures/f'{name}.png');plt.close(fig)
    for name,regime,metric,ylabel,title in [
        ('id_rollout','id_test','relative_l2','Physical relative L2 (%)','ID autoregressive error'),
        ('id_mass','id_test','relative_mass_error','Relative target mass error (%)','ID mass conservation diagnostic'),
        ('high_velocity','high_velocity','relative_l2','Physical relative L2 (%)','High-velocity autoregressive error')]:
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        for variant in LABELS:
            r=results[variant][regime]
            ax.plot(r['times'],100*np.asarray(r['summary']['autoregressive'][metric]['mean']),label=LABELS[variant],color=COLORS[variant],marker='o',markersize=3)
        ax.set(xlabel='Physical time',ylabel=ylabel,title=title);ax.legend();save(fig,name)
    comparison=[v for v in selection['validation_comparison'] if v['lambda_mass']>0]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric,label in zip(axes,['final_relative_l2','final_mass_error'],['Final validation AR L2 (%)','Final validation mass error (%)']):
        ax.plot([v['lambda_mass'] for v in comparison],[100*v['validation'][metric] for v in comparison],marker='o',color='#d97706')
        chosen=next(v for v in comparison if v['lambda_mass']==selection['C']['lambda_mass'])
        ax.scatter([chosen['lambda_mass']],[100*chosen['validation'][metric]],marker='*',s=150,color='#2563eb',label='Selected C')
        ax.set(xscale='log',xlabel='Mass-loss weight',ylabel=label);ax.legend()
    fig.suptitle('Validation-only lambda study: each candidate’s field-selected checkpoint');save(fig,'lambda_tradeoff')
    from dataset_generation import load_dataset
    data=load_dataset('data/dev/trajectories.npz');ids=results['A']['id_test']['trajectory_ids']
    predictions={}
    for variant in LABELS:
        with np.load(run/f'evaluation/{variant}/id_test_predictions.npz') as archive:predictions[variant]=archive['autoregressive']
    # Same baseline-defined worst trajectory for A/B/C: no per-model cherry-picking.
    worst=int(np.argmax(np.asarray(results['A']['id_test']['metrics']['autoregressive']['relative_l2'])[:,-1]))
    for tag,index in [('representative',0),('baseline_worst_id',worst)]:
        truth=data['fields'][ids[index],-1]
        fields=[truth]+[predictions[v][index,-1] for v in LABELS]
        low=min(f.min() for f in fields);high=max(f.max() for f in fields)
        limit=max(np.abs(f-truth).max() for f in fields[1:])
        fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
        for j,(field,label) in enumerate(zip(fields,['Solver reference',*LABELS.values()])):
            im=axes[0,j].imshow(field.T,origin='lower',extent=(0,2*np.pi,0,2*np.pi),vmin=low,vmax=high,cmap='viridis')
            axes[0,j].set(title=label,xlabel='x',ylabel='y');fig.colorbar(im,ax=axes[0,j],shrink=.8)
            if j==0:axes[1,j].axis('off');continue
            im=axes[1,j].imshow((field-truth).T,origin='lower',extent=(0,2*np.pi,0,2*np.pi),vmin=-limit,vmax=limit,cmap='RdBu_r')
            axes[1,j].set(title='Prediction − reference',xlabel='x',ylabel='y');fig.colorbar(im,ax=axes[1,j],shrink=.8)
        fig.suptitle(f'ID trajectory {ids[index]}, t=1 — {tag.replace("_"," ")}');save(fig,tag)
    print(public);print(public.with_name('stage5_ablation.csv'));print(figures)
    return output
