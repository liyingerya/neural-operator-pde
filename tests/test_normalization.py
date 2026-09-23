import numpy as np
from neural_operator.normalization import Normalization


def test_training_only_and_round_trip():
    rng=np.random.default_rng(1)
    data={'fields':rng.normal(size=(6,3,4,4)), 'coefficients':rng.normal(size=(6,3))}
    original={k:v.copy() for k,v in data.items()}
    norm=Normalization.fit(data,[0,2])
    assert norm.field_mean == data['fields'][[0,2]].mean()
    assert norm.field_std == data['fields'][[0,2]].std()
    np.testing.assert_allclose(norm.coefficient_mean,data['coefficients'][[0,2]].mean(0))
    np.testing.assert_allclose(norm.coefficient_std,data['coefficients'][[0,2]].std(0))
    for key in data:
        np.testing.assert_array_equal(data[key],original[key])
        data[key][[1,3,4,5]] = 1e10
    assert norm == Normalization.fit(data,[0,2])
    np.testing.assert_allclose(norm.inverse_field(norm.field(original['fields'])),original['fields'],atol=1e-14)


def test_constant_statistics_are_finite():
    data={'fields':np.ones((2,2,4,4)), 'coefficients':np.ones((2,3))}
    norm=Normalization.fit(data,[0])
    assert np.isfinite(norm.field(data['fields'])).all()
    assert np.isfinite(norm.coefficients(data['coefficients'])).all()
