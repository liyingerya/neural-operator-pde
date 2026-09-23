"""Float64 physical diagnostics and trajectory-wise aggregation."""
import numpy as np


def physical_diagnostics(prediction, target, initial, epsilon=1e-12):
    """Inputs: (trajectory,time,x,y), initial: (trajectory,x,y)."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    if prediction.shape != target.shape or initial.shape != target[:, 0].shape:
        raise ValueError('Diagnostic shapes disagree')
    area = (2*np.pi/prediction.shape[-1])**2
    axes = (-2, -1)
    mass = prediction.sum(axis=axes)*area
    target_mass = target.sum(axis=axes)*area
    initial_mass = initial.sum(axis=axes)*area
    initial_absolute_mass = np.abs(initial).sum(axis=axes)*area
    denominator = np.maximum(np.abs(initial_mass[:, None]), epsilon)
    finite = np.isfinite(prediction).all(axis=axes)
    with np.errstate(invalid='ignore', over='ignore'):
        l2 = np.sqrt(np.sum((prediction-target)**2, axis=axes)) / np.maximum(
            np.sqrt(np.sum(target**2, axis=axes)), epsilon)
        cumulative = np.concatenate((np.zeros((len(mass),1)),
                                     np.cumsum(np.abs(np.diff(mass, axis=1)), axis=1)), axis=1)/denominator
        negative_tolerance = 1e-7*np.maximum(1.0, np.max(np.abs(initial), axis=axes))[:, None, None, None]
        metrics = {
            'relative_l2': l2,
            'mass': mass,
            'relative_mass_error': np.abs(mass-target_mass)/np.maximum(np.abs(target_mass),epsilon),
            'signed_mass_drift': (mass-initial_mass[:,None])/denominator,
            'absolute_mass_drift': np.abs(mass-initial_mass[:,None])/denominator,
            'accumulated_absolute_mass_change': cumulative,
            'minimum': prediction.min(axis=axes), 'maximum': prediction.max(axis=axes),
            'negative_fraction': (prediction < 0).mean(axis=axes),
            'negative_fraction_below_tolerance': (prediction < -negative_tolerance).mean(axis=axes),
            'negative_mass_fraction': np.maximum(-prediction,0).sum(axis=axes)*area /
                                      np.maximum(initial_absolute_mass[:,None],epsilon),
        }
    for key, values in metrics.items():
        metrics[key] = np.where(finite, values, np.nan)
    return metrics


def feedback_amplification(autoregressive, teacher_forced, epsilon=1e-10):
    valid = np.isfinite(autoregressive) & np.isfinite(teacher_forced) & (teacher_forced > epsilon)
    return np.divide(autoregressive, teacher_forced, out=np.full_like(autoregressive,np.nan), where=valid)


def safe_list(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return float(values) if np.isfinite(values) else None
    return [safe_list(v) for v in values]


def summarize(metrics):
    """A horizon mean is unavailable if any trajectory failed, never silently omitted."""
    result = {}
    for name, values in metrics.items():
        valid = np.isfinite(values).all(axis=0)
        stats = {}
        for label, operation in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:
            with np.errstate(invalid='ignore'):
                stats[label] = safe_list(np.where(valid,operation(values,axis=0),np.nan))
        stats['finite_trajectory_count'] = np.isfinite(values).sum(axis=0).tolist()
        result[name] = stats
    return result
