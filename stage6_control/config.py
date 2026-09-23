"""Predeclared protocol; no settings are selected on held-out data."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 5e-4
    seed: int = 2028
    loader_seed: int = 5029
    threads: int = 4
    gate_updates: int = 300

    def to_dict(self):
        return asdict(self)


INTERPRETATION = {
    'case_1': 'D substantially outperforms B/C and approaches A: supports an accuracy tradeoff from the Stage 5 rollout objective/training setup under this protocol.',
    'case_2': 'D performs similarly poorly to B/C: degradation is more plausibly dominated by the shared lower-learning-rate/budget protocol rather than rollout training itself.',
    'case_3': 'D falls between A and B/C: mixed evidence; both training budget and objective may matter.',
    'operational_rule': 'Use held-out ID final AR mean L2. Case 1 first: D <= 0.8*min(B,C) and D <= 1.2*A. Case 2 next: abs(D/mean(B,C)-1) <= 0.1 and D > 1.2*A. Case 3 next: A < D < min(B,C). Otherwise report outside these cases, without changing thresholds.',
    'limits': 'Descriptive thresholds, not significance tests. One seed; unequal optimizer updates, horizon multiplicities, and gradient paths remain confounds.',
}


def classify(a, b, c, d):
    if d <= 0.8*min(b,c) and d <= 1.2*a:
        return 'case_1'
    if abs(d/((b+c)/2)-1) <= 0.1 and d > 1.2*a:
        return 'case_2'
    if a < d < min(b,c):
        return 'case_3'
    return 'outside_predeclared_cases'


def budget(epoch, examples=480, steps=1, batch_size=8):
    if epoch < 0 or examples % batch_size:
        raise ValueError('Expected nonnegative epochs and complete batches')
    return {'epoch':epoch, 'optimizer_updates':epoch*(examples//batch_size),
            'supervised_fields':epoch*examples*steps}
