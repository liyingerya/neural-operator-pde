import numpy as np
import pytest
from stage4_evaluation.diagnostics import physical_diagnostics,feedback_amplification,summarize


def test_mass_and_accumulation():
    initial=np.ones((1,2,2));target=np.ones((1,3,2,2))
    prediction=target*np.array([1,2,1])[None,:,None,None]
    d=physical_diagnostics(prediction,target,initial)
    np.testing.assert_allclose(d['signed_mass_drift'],[[0,1,0]])
    np.testing.assert_allclose(d['relative_mass_error'],[[0,1,0]])
    np.testing.assert_allclose(d['accumulated_absolute_mass_change'],[[0,1,2]])
    np.testing.assert_allclose(d['mass'][0,0],4*np.pi**2)


def test_negative_values_are_measured_without_correction():
    initial=np.ones((1,2,2));pred=initial[:,None].copy();pred[0,0,0,0]=-1
    original=pred.copy();d=physical_diagnostics(pred,initial[:,None],initial)
    assert d['negative_fraction'][0,0]==0.25
    assert d['negative_mass_fraction'][0,0]==0.25
    assert d['minimum'][0,0]==-1
    np.testing.assert_array_equal(pred,original)


def test_feedback_guard_and_population_aggregation():
    ar=np.array([[0.,1.,2.],[0.,3.,6.]])
    tf=np.array([[0.,1.,1e-12],[0.,1.,2.]])
    ratios=feedback_amplification(ar,tf)
    assert np.isnan(ratios[:,0]).all() and np.isnan(ratios[0,2])
    assert ratios[1,2]==3
    summary=summarize({'error':ar})
    assert summary['error']['mean']==[0,2,4]
    assert summary['error']['std']==[0,1,2]
