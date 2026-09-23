"""Build final tables and five figures using frozen public sources only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

BASE_REVISION='cfb580e134f8e30de3c6659b3ec29cf234c07cb2'
SOURCES=['docs/results/stage4_summary.json','docs/results/stage6_summary.json',
         'docs/stage3_fno.md','tests/test_solver.py']
LABELS={'A':'Historical one-step','B':'Rollout','C':'Rollout + mass','D':'Matched one-step'}
COLORS={'A':'#52616b','B':'#2771b6','C':'#b86419','D':'#16836b'}
FIGURES=['numerical_verification','rollout_error','ood_comparison','stage6_ablation','resolution_transfer']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_summary(root=Path('.')):
    root=Path(root)
    stage4=json.loads((root/SOURCES[0]).read_text())
    stage6=json.loads((root/SOURCES[1]).read_text())
    # Preserve the published rounded Stage 3 value, not a new persistence evaluation.
    line=next(line for line in (root/SOURCES[2]).read_text().splitlines() if line.startswith('| Persistence test |'))
    one_step_persistence=float(line.split('|')[2].strip())
    models={}
    for variant,model in stage6['models'].items():
        r=model['regimes'];idr=r['id_test'];ar=idr['horizon_means']['autoregressive']
        models[variant]={
            'objective':{'A':'one-step (historical)','B':'autoregressive K=5','C':'autoregressive K=5 + mass lambda=0.01','D':'one-step (matched control)'}[variant],
            'supervised_fields_full_run':model['supervised_fields_full_run'],
            'selected_epoch':model['selected_epoch'],
            'supervised_fields_selected':model['supervised_fields_at_selected_epoch'],
            'id_one_step_l2':idr['one_step_mean_relative_l2'],
            'id_final_ar_l2':ar['relative_l2'][-1],
            'id_mass_error':ar['relative_mass_error'][-1],
            'id_negative_mass_fraction':ar['negative_mass_fraction'][-1],
            'fresh_id_final_ar_l2':r['fresh_id']['horizon_means']['autoregressive']['relative_l2'][-1],
            'high_nu_final_ar_l2':r['high_nu']['horizon_means']['autoregressive']['relative_l2'][-1],
            'high_velocity_final_ar_l2':r['high_velocity']['horizon_means']['autoregressive']['relative_l2'][-1],
            'native_64_final_ar_l2':r['resolution_64']['horizon_means']['autoregressive']['relative_l2'][-1],
            'transferred_128_final_ar_l2':r['resolution_128']['horizon_means']['autoregressive']['relative_l2'][-1]}
    idr=stage4['regimes']['id_test']
    return {'schema_version':1,'frozen_source_revision':BASE_REVISION,
        'source_sha256':{p:digest(root/p) for p in SOURCES},
        'units':'All errors/fractions are dimensionless; plots and human tables multiply by 100.',
        'count_note':'Full training budgets exclude discarded gates and C candidate search cost; selected-checkpoint counts are separate. Equal field exposure is not equal update count or exact compute.',
        'models':models,'persistence':{'one_step_id_l2_published_rounded':one_step_persistence,
            'one_step_definition':'Copy the true field at each preceding snapshot; Stage 3 float32-target evaluation.',
            'final_id_rollout_l2':idr['horizon_means']['persistence']['relative_l2'][-1],
            'rollout_definition':'Keep the initial field u0 at every horizon; Stage 4 float64 reference evaluation.'},
        'baseline_id_horizons':{'times':idr['times'],**{name:idr['horizon_means'][name]['relative_l2'] for name in ['teacher_forced','autoregressive','persistence']}},
        'cpu_id_benchmark':stage4['benchmark']['id_test'],
        'cpu_device':stage4['device'],
        'interpretation_case':stage6['interpretation_case'],
        'figure_sources':{name:('Frozen solver tests: verification diagram, not measured convergence curves' if name=='numerical_verification' else 'Frozen public Stage 4/6 summaries; no inference') for name in FIGURES}}


def write_tables(summary,output):
    results=output/'results';results.mkdir(parents=True,exist_ok=True)
    (results/'final_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (results/'final_model_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model',*next(iter(summary['models'].values()))],lineterminator='\n')
        writer.writeheader()
        for variant,row in summary['models'].items():writer.writerow({'model':variant,**row})


def plot_figures(summary,output):
    os.environ['MPLCONFIGDIR']=str(Path('runs/stage7/mplconfig').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':150,'font.family':'DejaVu Sans'})
    dest=output/'figures';dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/f'{name}.png',facecolor='white',metadata={'Software':'Frozen project synthesis'})
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.4),layout='constrained');ax.axis('off')
    rows=[['Centered differences','Grid refinement','Second-order spatial'],
          ['RK4 time stepping','Discrete Fourier reference','Fourth-order temporal'],
          ['Fourier-mode evolution','Analytic mode solution','Advection / diffusion'],
          ['Periodic differences','Constant fields and sums','Conservation']]
    table=ax.table(cellText=rows,colLabels=['Numerical foundation','Verification method','Property checked'],
                   cellLoc='left',colLoc='left',loc='center',colWidths=[.32,.38,.30])
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2.2)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d2d6d9')
        if row==0:cell.set_facecolor('#edf1f3');cell.set_text_props(weight='bold')
    ax.set_title('Numerical reference: verification before learning',pad=12)
    fig.text(.5,.015,'Map of frozen regression checks; not a newly measured convergence curve.',ha='center',fontsize=10)
    save(fig,'numerical_verification')
    h=summary['baseline_id_horizons']
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
    for key,label,color in [('teacher_forced','Teacher-forced one-step','#16836b'),('autoregressive','Autoregressive','#2771b6'),('persistence','Persistence: retain initial field','#52616b')]:
        ax.plot(h['times'],100*np.asarray(h[key]),label=label,color=color,marker='o',markersize=3)
    ax.set(xlabel='Physical time',ylabel='Mean relative L2 error (%)',title='Historical baseline A: one-step accuracy does not ensure rollout accuracy')
    ax.legend(loc='upper left',fontsize=10);save(fig,'rollout_error')
    models=summary['models'];names=['id_final_ar_l2','fresh_id_final_ar_l2','high_nu_final_ar_l2','high_velocity_final_ar_l2']
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained');x=np.arange(4)
    for offset,v in [(-.18,'A'),(.18,'D')]:
        bars=ax.bar(x+offset,[100*models[v][k] for k in names],width=.34,label=LABELS[v],color=COLORS[v])
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
    ax.set(xticks=x,xticklabels=['Original ID','Fresh ID','High diffusivity','High velocity'],ylabel='Final rollout relative L2 (%)',ylim=(0,49),title='Parameter extrapolation depends on the regime')
    ax.legend(fontsize=10);save(fig,'ood_comparison')
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,['id_one_step_l2','id_final_ar_l2','id_mass_error'],['ID one-step error','ID final rollout error','ID final mass error']):
        bars=ax.bar(list(models),[100*m[key] for m in models.values()],color=[COLORS[v] for v in models])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
        ax.set(title=title,ylabel='Relative error (%)',ylim=(0,max(100*m[key] for m in models.values())*1.22))
    fig.suptitle('A: historical   B: rollout   C: rollout + mass   D: matched one-step',fontsize=12)
    save(fig,'stage6_ablation')
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
    x=np.arange(len(models))
    for offset,key,label,color in [(-.18,'native_64_final_ar_l2','64 × 64','#52616b'),(.18,'transferred_128_final_ar_l2','128 × 128','#88aabd')]:
        bars=ax.bar(x+offset,[100*m[key] for m in models.values()],width=.34,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
    ax.set(xticks=x,xticklabels=list(models),xlabel='Model (same weights and normalization on both grids)',ylabel='Final native-grid relative L2 (%)',ylim=(0,31),title='Matched grid transfer: errors against discrete solver references')
    ax.legend(ncol=2,fontsize=10,loc='upper right');save(fig,'resolution_transfer')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('docs'))
    args=parser.parse_args();summary=build_summary()
    write_tables(summary,args.output_root);plot_figures(summary,args.output_root)
    print('Built final JSON/CSV and five figures from frozen public sources only.')


if __name__=='__main__':main()
