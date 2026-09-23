"""Static scientific figures built only from saved evaluation results."""
import json
import os
from pathlib import Path
import numpy as np


def make_figures(directory=Path('runs/stage4')):
    directory=Path(directory)
    os.environ.setdefault('MPLCONFIGDIR',str((directory/'mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':140,'savefig.dpi':160})
    output=directory/'figures';output.mkdir(exist_ok=True)
    names=['id_test','fresh_id','high_nu','high_velocity','velocity_pp','velocity_pn','velocity_np','velocity_nn','resolution_64','resolution_128']
    results={name:json.loads((directory/f'{name}.json').read_text()) for name in names}
    methods={'teacher_forced':('Teacher-forced','#d97706'),'autoregressive':('Autoregressive','#2563eb'),'persistence':('Persistence rollout','#6b7280')}
    def save(fig,name):
        fig.savefig(output/f'{name}.png');plt.close(fig)
    def curve(result,method,metric):
        return np.asarray(result['summary'][method][metric]['mean'],dtype=float)
    times=np.asarray(results['id_test']['times'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method,(label,color) in methods.items():
        axes[0].plot(times[1:],100*curve(results['id_test'],method,'relative_l2')[1:],label=label,color=color,marker='o',markersize=3)
    axes[0].set(xlabel='Physical time',ylabel='Mean relative L2 (%)',title='ID: one-step inputs versus feedback');axes[0].legend()
    for row in results['id_test']['metrics']['autoregressive']['relative_l2']:
        axes[1].plot(times,100*np.asarray(row),color='#2563eb',alpha=.2)
    axes[1].plot(times,100*curve(results['id_test'],'autoregressive','relative_l2'),color='#2563eb',linewidth=2,label='Mean of 8 trajectories')
    axes[1].set(xlabel='Physical time',ylabel='Relative L2 (%)',title='ID autoregressive trajectory spread');axes[1].legend()
    save(fig,'id_rollout_error')
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name in ['id_test','fresh_id','high_nu','high_velocity']:
        ax.plot(times[1:],results[name]['feedback_amplification']['ratio_of_mean_errors'][1:],label=name,marker='o',markersize=3)
    ax.set(xlabel='Physical time',ylabel='AR mean error / teacher-forced mean error',title='Feedback amplification (diagnostic, not a stability bound)');ax.legend()
    save(fig,'feedback_amplification')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    metrics=[('relative_mass_error','Mass error against target (%)'),('signed_mass_drift','Signed mass drift from initial (%)'),
             ('negative_fraction','Negative cells (%)'),('negative_mass_fraction','Integrated negative fraction (%)')]
    for ax,(metric,label) in zip(axes.flat,metrics):
        for name in ['id_test','fresh_id','high_nu','high_velocity']:
            ax.plot(times,100*curve(results[name],'autoregressive',metric),label=name)
        ax.set(xlabel='Physical time',ylabel=label)
    axes[0,0].legend();fig.suptitle('Vanilla FNO physical diagnostics; no corrections')
    save(fig,'physical_diagnostics')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    regimes=['id_test','fresh_id','high_nu','high_velocity']
    for ax,metric_method,title in [(axes[0],'teacher_forced','Teacher-forced: mean over ten horizons'),(axes[1],'autoregressive','Autoregressive: final time 1.0')]:
        values=[]
        for name in regimes:
            v=curve(results[name],metric_method,'relative_l2')
            values.append(100*(v[1:].mean() if metric_method=='teacher_forced' else v[-1]))
        ax.bar(regimes,values,color=['#2563eb','#0891b2','#d97706','#9333ea']);ax.set(ylabel='Relative L2 (%)',title=title)
    save(fig,'id_ood_comparison')
    benchmark=json.loads((directory/'benchmark.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,batch in zip(axes,['1','8']):
        timing=benchmark['workloads']['id_test'][batch]
        labels=['solver','fno_forward','fno_end_to_end'];med=np.array([timing[k]['median_seconds'] for k in labels])*1000
        lower=med-np.array([timing[k]['q25_seconds'] for k in labels])*1000
        upper=np.array([timing[k]['q75_seconds'] for k in labels])*1000-med
        ax.bar(['Solver','FNO forward','FNO incl. transforms'],med,yerr=[lower,upper],capsize=4,color=['#6b7280','#93c5fd','#2563eb'])
        ax.set(ylabel='CPU workload latency (ms)',title=f'ID batch {batch}; median and IQR')
    fig.suptitle('Float64 sequential solver versus float32 FNO — not equal-accuracy methods')
    save(fig,'cpu_timing')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for n,color in [(64,'#2563eb'),(128,'#d97706')]:
        r=results[f'resolution_{n}']
        axes[0].plot(times[1:],100*curve(r,'teacher_forced','relative_l2')[1:],label=f'{n}×{n}',color=color)
        axes[1].plot(times,100*curve(r,'autoregressive','relative_l2'),label=f'{n}×{n}',color=color)
    for ax,title in zip(axes,['Matched physical problems: teacher forcing','Matched physical problems: autoregression']):
        ax.set(xlabel='Physical time',ylabel='Native-grid relative L2 (%)',title=title);ax.legend()
    save(fig,'resolution_transfer')
    from dataset_generation import load_dataset
    data=load_dataset('data/dev/trajectories.npz')
    r=results['id_test']; ids=r['trajectory_ids']
    with np.load(directory/'id_test_predictions.npz') as archive:
        predictions=archive['autoregressive']
    worst=int(np.argmax(np.asarray(r['metrics']['autoregressive']['relative_l2'])[:,-1]))
    for tag,index in [('predetermined',0),('worst_final_error',worst)]:
        truth=data['fields'][ids[index]];prediction=predictions[index];steps=[1,5,10]
        low=min(truth[steps].min(),prediction[steps].min());high=max(truth[steps].max(),prediction[steps].max())
        error=prediction-truth;limit=np.max(np.abs(error[steps]))
        fig,axes=plt.subplots(3,3,figsize=(10,9),layout='constrained')
        for col,k in enumerate(steps):
            for row,(array,title) in enumerate([(truth,'Numerical reference'),(prediction,'Autoregressive FNO'),(error,'Prediction − reference')]):
                im=axes[row,col].imshow(array[k].T,origin='lower',extent=(0,2*np.pi,0,2*np.pi),
                    vmin=-limit if row==2 else low,vmax=limit if row==2 else high,cmap='RdBu_r' if row==2 else 'viridis')
                axes[row,col].set(title=f'{title}, t={times[k]:.1f}',xlabel='x',ylabel='y');fig.colorbar(im,ax=axes[row,col],shrink=.8)
        fig.suptitle(f'ID trajectory {ids[index]} — {tag.replace("_"," ")}')
        save(fig,f'fields_{tag}')
    return sorted(str(p) for p in output.glob('*.png'))
