"""Aggregate the frozen four-model evaluation and expose budget differences."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from stage4_evaluation.checkpoint import write_json,sha256,verify_hash
from stage5_training.reporting import REGIMES
from .config import classify,budget,INTERPRETATION

LABELS={'A':'Historical one-step','B':'Rollout','C':'Rollout + mass','D':'Matched one-step'}
COLORS={'A':'#64748b','B':'#2563eb','C':'#d97706','D':'#059669'}


def budget_curves(run):
    curves={};source_hashes={}
    for variant in ('B','C'):
        selected=json.loads(Path('runs/stage5/selection.json').read_text())[variant]['directory']
        # Follow the frozen selected C directory; never choose it using new results.
        path=Path('runs/stage5')/selected/'history.json'
        rows=json.loads(path.read_text());elapsed=0.;curve=[]
        for row in rows:
            elapsed+=row['runtime_seconds']
            curve.append({**budget(row['epoch'],288,5),'cumulative_epoch_seconds':elapsed,
                'validation_final_ar':row['validation']['final_relative_l2'],
                'training_field_loss':row['training']['field']})
        curves[variant]=curve;source_hashes[str(path)]=sha256(path)
    path=Path(run)/'history.json';rows=json.loads(path.read_text())
    curves['D']=[{k:row[k] for k in ['epoch','optimizer_updates','supervised_fields','cumulative_epoch_seconds']}
        | {'validation_final_ar':row['validation']['final_relative_l2'],
           'validation_one_step':row['validation']['one_step_relative_l2'],'training_field_loss':row['training_loss'],
           'cumulative_training_seconds':row['cumulative_training_seconds']} for row in rows]
    source_hashes[str(path)]=sha256(path)
    comparisons={}
    for axis in ['epoch','optimizer_updates','supervised_fields']:
        by_variant={v:{r[axis]:r['validation_final_ar'] for r in rows} for v,rows in curves.items()}
        common=sorted(set.intersection(*(set(rows) for rows in by_variant.values())))
        comparisons[axis]=[{axis:x,**{v:rows[x] for v,rows in by_variant.items()}} for x in common]
    return curves,comparisons,source_hashes


def report(run,public):
    run=Path(run);public=Path(public)
    protocol=json.loads((run/'protocol.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    evaluation=json.loads((run/'evaluation/manifest.json').read_text())
    verify_hash(run/'selection.json',evaluation['selection_sha256'])
    if protocol['interpretation']!=INTERPRETATION:
        raise ValueError('Interpretation changed after predeclaration')
    raw={v:{r:json.loads((run/f'evaluation/{v}/{r}.json').read_text()) for r in REGIMES} for v in LABELS}
    curves,comparisons,history_hashes=budget_curves(run)
    previous=json.loads(Path('runs/stage5/selection.json').read_text())
    selected_epochs={'A':31,'B':previous['B']['epoch'],'C':previous['C']['epoch'],'D':selection['epoch']}
    fields_per_epoch={'A':480,'B':1440,'C':1440,'D':480}
    full_epochs={'A':100,'B':20,'C':20,'D':60}
    gates=json.loads((run/'gate.json').read_text())
    output={'protocol':protocol,'selection':selection,'gate':{k:v for k,v in gates.items() if k!='history'},
        'optional_e':json.loads((run/'optional_e.json').read_text()),'evaluation_manifest':evaluation,
        'budget_curves':curves,'exact_budget_comparisons':comparisons,'history_sha256':history_hashes,
        'budget_time_definition':'Sum of epoch training+validation wall seconds; B/C optimization-only timings were not separately recorded. Gates, evaluation, and most checkpoint I/O excluded.',
        'models':{},'source_sha256':{str(p):sha256(p) for p in sorted(Path('stage6_control').glob('*.py'))+sorted(Path('examples').glob('*stage6.py'))}}
    for variant,regimes in raw.items():
        model={'label':LABELS[variant],'selected_epoch':selected_epochs[variant],
               'supervised_fields_full_run':full_epochs[variant]*fields_per_epoch[variant],
               'supervised_fields_at_selected_epoch':selected_epochs[variant]*fields_per_epoch[variant],
               'regimes':{},'resolution_comparison':json.loads((run/f'evaluation/{variant}/resolution_comparison.json').read_text())}
        for name,r in regimes.items():
            model['regimes'][name]={'trajectory_count':len(r['trajectory_ids']),'times':r['times'],
                'one_step_mean_relative_l2':float(np.asarray(r['metrics']['teacher_forced']['relative_l2'])[:,1:].mean()),
                'horizon_means':{method:{metric:values[metric]['mean'] for metric in (values if method=='autoregressive' else ['relative_l2','relative_mass_error'])} for method,values in r['summary'].items()},
                'feedback_ratio_of_means':r['feedback_amplification']['ratio_of_mean_errors'],
                'minimum_across_trajectories':r['summary']['autoregressive']['minimum']['min'],
                'maximum_across_trajectories':r['summary']['autoregressive']['maximum']['max'],
                'failures':r['failures']}
        output['models'][variant]=model
    errors=[output['models'][v]['regimes']['id_test']['horizon_means']['autoregressive']['relative_l2'][-1] for v in LABELS]
    output['interpretation_case']=classify(*errors)
    write_json(public,output)
    with public.with_name('stage6_comparison.csv').open('x',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['model','objective','supervised_fields_full_run','selected_epoch','supervised_fields_selected','id_one_step_l2','id_final_ar_l2','id_mass_error','high_velocity_ar','resolution_64_ar','resolution_128_ar'])
        for v,m in output['models'].items():
            r=m['regimes'];a=r['id_test']['horizon_means']['autoregressive']
            writer.writerow([v,{'A':'one-step','B':'rollout K=5','C':'rollout K=5 + mass lambda=0.01','D':'one-step'}[v],m['supervised_fields_full_run'],m['selected_epoch'],m['supervised_fields_at_selected_epoch'],
                r['id_test']['one_step_mean_relative_l2'],a['relative_l2'][-1],a['relative_mass_error'][-1],
                *[r[k]['horizon_means']['autoregressive']['relative_l2'][-1] for k in ['high_velocity','resolution_64','resolution_128']]])
    figures=run/'figures';figures.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str((run/'mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,key,label in zip(axes.flat,['epoch','optimizer_updates','supervised_fields','cumulative_epoch_seconds'],['Epoch','Optimizer updates','Supervised predicted fields','Cumulative training + validation seconds']):
        for v,rows in curves.items():
            ax.plot([r[key] for r in rows],[100*r['validation_final_ar'] for r in rows],label=LABELS[v],color=COLORS[v])
        ax.set(xlabel=label,ylabel='Validation final AR L2 (%)');ax.legend()
    fig.suptitle('Budget comparison: saved per-epoch validation measurements')
    fig.savefig(figures/'budget_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key,label in zip(axes,['relative_l2','relative_mass_error'],['ID AR relative L2 (%)','ID target mass error (%)']):
        for v,m in output['models'].items():
            r=m['regimes']['id_test'];ax.plot(r['times'],100*np.asarray(r['horizon_means']['autoregressive'][key]),label=LABELS[v],color=COLORS[v])
        ax.set(xlabel='Physical time',ylabel=label);ax.legend()
    fig.savefig(figures/'id_comparison.png');plt.close(fig)
    print(public,public.with_name('stage6_comparison.csv'),figures,sep='\n')
    return output
