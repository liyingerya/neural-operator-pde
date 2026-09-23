import pytest
torch=pytest.importorskip('torch')
from neural_operator.fno import FNO2d, parameter_counts, SpectralConv2d
from neural_operator.training import seed_everything


@pytest.mark.parametrize('batch',[1,2])
def test_forward_and_backward(batch):
    seed_everything()
    model=FNO2d()
    x=torch.randn(batch,4,64,64)
    y=model(x)
    assert y.shape == (batch,1,64,64)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.spectral[0].positive.grad.abs().sum() > 0
    assert model.spectral[0].negative.grad.abs().sum() > 0


def test_periodic_shift_equivariance():
    seed_everything()
    model=FNO2d(width=4,blocks=2,modes=3,projection_width=8)
    x=torch.randn(2,4,16,16)
    shifted=model(torch.roll(x,(3,-2),(-2,-1)))
    torch.testing.assert_close(shifted,torch.roll(model(x),(3,-2),(-2,-1)),atol=2e-6,rtol=2e-5)


def test_invalid_modes_and_counts():
    with pytest.raises(ValueError):
        SpectralConv2d(4,12)(torch.randn(1,4,16,16))
    counts=parameter_counts(FNO2d())
    assert counts['trainable_parameter_elements'] == 1186209
    assert counts['trainable_real_scalar_parameters'] == 2365857
