"""Teacher forcing and feedback use the same frozen one-step operator."""
import numpy as np
import torch
from .diagnostics import physical_diagnostics, feedback_amplification, summarize, safe_list


def prepare_input(fields, coefficients, normalization):
    field = torch.as_tensor(normalization.field(np.asarray(fields)), dtype=torch.float32)[:,None]
    condition = torch.as_tensor(normalization.coefficients(coefficients), dtype=torch.float32)
    return torch.cat((field, condition[:,:,None,None].expand(-1,-1,*field.shape[-2:])), dim=1)


@torch.inference_mode()
def predict_trajectories(model, normalization, fields, coefficients, times, batch_size=8):
    fields = np.asarray(fields)
    m,s,n,n2 = fields.shape
    if n != n2 or coefficients.shape != (m,3) or times.shape != (s,):
        raise ValueError('Invalid trajectory shapes')
    if not np.allclose(np.diff(times),0.1,rtol=0,atol=1e-12):
        raise ValueError('Frozen model requires 0.1 intervals')
    if not np.isfinite(fields).all() or not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite numerical references')
    model.eval()
    outputs = {name: np.full(fields.shape,np.nan,dtype=np.float64)
               for name in ('teacher_forced','autoregressive')}
    for value in outputs.values():
        value[:,0] = fields[:,0]
    for start in range(0,m,batch_size):
        stop = min(start+batch_size,m)
        current = prepare_input(fields[start:stop,0],coefficients[start:stop],normalization)
        condition = current[:,1:].clone()
        for k in range(1,s):
            teacher = model(prepare_input(fields[start:stop,k-1],coefficients[start:stop],normalization))
            prediction = model(current)
            outputs['teacher_forced'][start:stop,k] = normalization.inverse_field(teacher[:,0].double()).numpy()
            outputs['autoregressive'][start:stop,k] = normalization.inverse_field(prediction[:,0].double()).numpy()
            current = torch.cat((prediction,condition),dim=1)
    # A failed trajectory is unavailable from its first nonfinite prediction onward.
    failures = {}
    for name, values in outputs.items():
        failures[name] = []
        for i in range(m):
            bad = np.flatnonzero(~np.isfinite(values[i]).all(axis=(-2,-1)))
            if len(bad):
                first = int(bad[0]); values[i,first:] = np.nan
                failures[name].append({'trajectory_index':i,'first_failure_step':first})
    outputs['persistence'] = np.broadcast_to(fields[:,:1],fields.shape).copy()
    return outputs, failures


def evaluate_archive(model, normalization, data, archive_id, ids=None):
    ids = list(range(len(data['fields']))) if ids is None else list(ids)
    fields = data['fields'][ids]
    outputs, failures = predict_trajectories(model,normalization,fields,data['coefficients'][ids],data['times'])
    metrics = {name:physical_diagnostics(values,fields,fields[:,0]) for name,values in outputs.items()}
    metrics['reference'] = physical_diagnostics(fields,fields,fields[:,0])
    feedback = feedback_amplification(metrics['autoregressive']['relative_l2'],metrics['teacher_forced']['relative_l2'])
    result = {'archive_id':archive_id,'trajectory_ids':ids,'times':data['times'].tolist(),
              'grid_size':int(data['grid_size']), 'failures':failures,
              'metrics':{name:{key:safe_list(value) for key,value in values.items()} for name,values in metrics.items()},
              'summary':{name:summarize(values) for name,values in metrics.items()},
              'feedback_amplification':{'denominator_threshold':1e-10,'per_trajectory':safe_list(feedback),
                  'summary':summarize({'ratio':feedback})['ratio']}}
    ar = np.asarray(result['summary']['autoregressive']['relative_l2']['mean'],dtype=float)
    tf = np.asarray(result['summary']['teacher_forced']['relative_l2']['mean'],dtype=float)
    result['feedback_amplification']['ratio_of_mean_errors'] = safe_list(feedback_amplification(ar,tf))
    return result, outputs


def combine_results(results, archive_id):
    """Combine sign-stratified archives without falsifying their Stage 2 metadata."""
    names = results[0]['metrics'].keys()
    metrics = {name:{key:np.concatenate([np.asarray(r['metrics'][name][key],dtype=float) for r in results])
                     for key in results[0]['metrics'][name]} for name in names}
    ratios = feedback_amplification(metrics['autoregressive']['relative_l2'],metrics['teacher_forced']['relative_l2'])
    return {'archive_id':archive_id,'times':results[0]['times'],'grid_size':results[0]['grid_size'],
            'trajectory_ids':[f"{r['archive_id']}:{i}" for r in results for i in r['trajectory_ids']],
            'failures':{r['archive_id']:r['failures'] for r in results},
            'metrics':{name:{key:safe_list(v) for key,v in values.items()} for name,values in metrics.items()},
            'summary':{name:summarize(values) for name,values in metrics.items()},
            'feedback_amplification':{'per_trajectory':safe_list(ratios),'summary':summarize({'ratio':ratios})['ratio'],
              'ratio_of_mean_errors':safe_list(feedback_amplification(metrics['autoregressive']['relative_l2'].mean(0),metrics['teacher_forced']['relative_l2'].mean(0))),
              'denominator_threshold':1e-10}}
