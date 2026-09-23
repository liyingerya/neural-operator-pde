import pytest
torch=pytest.importorskip('torch')
from neural_operator.normalization import Normalization
from stage5_training.objective import autoregressive_steps,losses
from stage5_training.config import validate_lambda


class Gain(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gain=torch.nn.Parameter(torch.tensor(2.));self.conditions=[]
    def forward(self,x):
        self.conditions.append(x[:,1:].detach().clone())
        return self.gain*x[:,:1]


def test_full_feedback_gradient_and_fixed_conditioning():
    model=Gain();x=torch.ones(1,4,2,2)
    outputs=autoregressive_steps(model,x,5)
    for value in outputs:value.retain_grad()
    outputs[-1].mean().backward()
    assert model.gain.grad.item()==pytest.approx(5*2**4)
    assert [v[0,0,0,0].item() for v in outputs]==[2,4,8,16,32]
    assert all(v.grad is not None and v.grad.abs().sum()>0 for v in outputs)
    assert all(torch.equal(c,model.conditions[0]) for c in model.conditions)


def test_mass_loss_is_physical_differentiable_and_unclipped():
    norm=Normalization(1,2,[0,0,0],[1,1,1])
    p=torch.zeros(2,1,2,2,requires_grad=True)
    initial_mass=torch.full((2,),8*torch.pi**2,dtype=torch.float64)
    value=losses([p]*5,torch.zeros(2,5,1,2,2),initial_mass,norm,0.1)
    assert value['mass'].item()==pytest.approx(0.25)
    assert value['total'].item()==pytest.approx(0.025)
    value['mass'].backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert p.grad.sum().item()==pytest.approx(-1.)


def test_equal_horizon_weighting_and_zero_mass_guard():
    norm=Normalization(0,1,[0,0,0],[1,1,1])
    predictions=[torch.full((1,1,2,2),float(i),requires_grad=True) for i in range(1,6)]
    value=losses(predictions,torch.zeros(1,5,1,2,2),torch.zeros(1,dtype=torch.float64),norm,0)
    assert value['field'].item()==pytest.approx(11.)
    assert torch.isfinite(value['mass'])
    assert value['total'].item()==pytest.approx(value['field'].item())


@pytest.mark.parametrize('value',[-1,0.02,float('nan'),10])
def test_lambda_restriction(value):
    with pytest.raises(ValueError):validate_lambda(value)
