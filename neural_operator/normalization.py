"""Global training-only affine normalization; never mutate archive arrays."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Normalization:
    field_mean: float
    field_std: float
    coefficient_mean: list[float]
    coefficient_std: list[float]

    @classmethod
    def fit(cls, data, train_ids):
        ids = np.asarray(train_ids, dtype=np.int64)
        if len(ids) == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError("Training IDs must be nonempty and unique")
        fields = data['fields'][ids].astype(np.float64)
        coefficients = data['coefficients'][ids].astype(np.float64)
        return cls(float(fields.mean()), max(float(fields.std()), 1e-12),
                   coefficients.mean(axis=0).tolist(),
                   np.maximum(coefficients.std(axis=0), 1e-12).tolist())

    def field(self, value):
        return (value - self.field_mean) / self.field_std

    def inverse_field(self, value):
        return value * self.field_std + self.field_mean

    def coefficients(self, value):
        return (value - np.asarray(self.coefficient_mean)) / np.asarray(self.coefficient_std)

    def to_dict(self):
        return asdict(self)
