"""Physical-unit metrics; mass error is a diagnostic, never a loss."""
import torch


def physical_errors(prediction, target, epsilon=1e-12):
    prediction, target = prediction.double().flatten(1), target.double().flatten(1)
    relative_l2 = torch.linalg.vector_norm(prediction-target, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target, dim=1), min=epsilon)
    relative_mass = (prediction.sum(1)-target.sum(1)).abs() / target.sum(1).abs().clamp_min(epsilon)
    return relative_l2, relative_mass


def aggregate(records):
    """Average pairs within trajectory, then trajectories with equal weight."""
    by_id = {}
    for trajectory, l2, mass in records:
        by_id.setdefault(int(trajectory), []).append((l2, mass))
    per_trajectory = {str(i): {'relative_l2': sum(v[0] for v in values)/len(values),
                             'relative_mass_error': sum(v[1] for v in values)/len(values)}
                      for i, values in sorted(by_id.items())}
    return {'relative_l2': sum(v['relative_l2'] for v in per_trajectory.values())/len(by_id),
            'relative_mass_error': sum(v['relative_mass_error'] for v in per_trajectory.values())/len(by_id),
            'per_trajectory': per_trajectory}
