import pytest
torch=pytest.importorskip('torch')
from neural_operator.metrics import physical_errors, aggregate


def test_physical_metrics():
    target=torch.ones(2,1,4,4)
    l2,mass=physical_errors(target,target)
    assert (l2==0).all() and (mass==0).all()
    l2,mass=physical_errors(2*target,target)
    torch.testing.assert_close(l2,torch.ones(2,dtype=torch.float64))
    torch.testing.assert_close(mass,torch.ones(2,dtype=torch.float64))
    l2,mass=physical_errors(target,torch.zeros_like(target))
    assert torch.isfinite(l2).all() and torch.isfinite(mass).all()


def test_equal_trajectory_weighting():
    result=aggregate([(1,0.1,0.2),(1,0.3,0.4),(2,0.6,0.8)])
    assert result['relative_l2'] == pytest.approx(0.4)
    assert result['relative_mass_error'] == pytest.approx(0.55)
