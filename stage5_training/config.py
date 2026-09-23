"""Predeclared CPU ablation protocol; no test/OOD-driven choices."""
from dataclasses import dataclass,asdict

LAMBDAS=(0.01,0.1,1.0)


@dataclass(frozen=True)
class TrainingConfig:
    steps:int=5
    epochs:int=20
    batch_size:int=8
    learning_rate:float=5e-4
    seed:int=2028
    loader_seed:int=5029
    threads:int=4
    gate_updates:int=300
    mass_epsilon:float=1e-12

    def to_dict(self):
        return asdict(self)


def validate_lambda(value):
    if value not in (0.0,*LAMBDAS):
        raise ValueError('Only lambda 0, 0.01, 0.1, or 1.0 is predeclared')
    return float(value)
