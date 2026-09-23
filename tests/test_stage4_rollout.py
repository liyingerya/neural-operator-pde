import numpy as np
import pytest
torch=pytest.importorskip('torch')
from neural_operator.normalization import Normalization
from stage4_evaluation.rollout import predict_trajectories,evaluate_archive,combine_results


class AddOne(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conditions=[]
    def forward(self,x):
        self.conditions.append(x[:,1:].clone())
        return x[:,:1]+1


def fixture():
    fields=np.ones((2,4,4,4))+2*np.arange(4)[None,:,None,None]
    data={'fields':fields,'coefficients':np.array([[0.1,0.2,0.03],[-0.2,0.4,0.02]]),
          'times':np.arange(4)*0.1,'grid_size':np.asarray(4)}
    return data,Normalization(0,1,[0,0,0],[1,1,1])


def test_feedback_alignment_and_fixed_coefficients(monkeypatch):
    data,norm=fixture(); model=AddOne()
    monkeypatch.setattr(Normalization,'fit',lambda *args:pytest.fail('Normalization must not be fitted'))
    outputs,failures=predict_trajectories(model,norm,data['fields'],data['coefficients'],data['times'])
    assert outputs['autoregressive'].shape==(2,4,4,4)
    np.testing.assert_array_equal(outputs['autoregressive'][0,:,0,0],[1,2,3,4])
    np.testing.assert_array_equal(outputs['teacher_forced'][0,:,0,0],[1,2,4,6])
    np.testing.assert_array_equal(outputs['persistence'][0,:,0,0],[1,1,1,1])
    for condition in model.conditions:
        torch.testing.assert_close(condition,model.conditions[0])
    assert not any(failures.values())


def test_original_float64_targets_and_denormalization():
    data,norm=fixture(); norm=Normalization(0.25,2,[0,0,0],[1,1,1])
    result,outputs=evaluate_archive(AddOne(),norm,data,'fixture')
    np.testing.assert_allclose(outputs['autoregressive'],data['fields'])
    assert result['summary']['autoregressive']['relative_l2']['mean']==[0.0]*4
    assert result['feedback_amplification']['ratio_of_mean_errors'][0] is None


def test_time_rejection():
    data,norm=fixture();data['times']*=2
    with pytest.raises(ValueError,match='0.1'):
        predict_trajectories(AddOne(),norm,data['fields'],data['coefficients'],data['times'])


def test_nonfinite_failure_is_not_hidden():
    class Bad(torch.nn.Module):
        def forward(self,x):
            y=x[:,:1].clone();y[0]=float('nan');return y
    data,norm=fixture()
    result,_=evaluate_archive(Bad(),norm,data,'fixture')
    assert result['failures']['autoregressive']==[{'trajectory_index':0,'first_failure_step':1}]
    assert result['summary']['autoregressive']['relative_l2']['mean'][1:] == [None]*3
    assert result['summary']['autoregressive']['relative_l2']['finite_trajectory_count']==[2,1,1,1]


def test_combined_archive_ids_and_aggregation():
    data,norm=fixture()
    a,_=evaluate_archive(AddOne(),norm,data,'a')
    b,_=evaluate_archive(AddOne(),norm,data,'b')
    combined=combine_results([a,b],'all')
    assert combined['trajectory_ids']==['a:0','a:1','b:0','b:1']
    for method in a['summary']:
        for metric in a['summary'][method]:
            assert combined['summary'][method][metric]['mean']==a['summary'][method][metric]['mean']
            assert combined['summary'][method][metric]['finite_trajectory_count']==[4]*4


def test_frozen_fno_accepts_both_native_grids():
    from neural_operator.fno import FNO2d
    torch.set_num_threads(4)
    model=FNO2d().eval().requires_grad_(False)
    with torch.inference_mode():
        for n in (64,128):
            output=model(torch.zeros(1,4,n,n))
            assert output.shape==(1,1,n,n) and torch.isfinite(output).all()


def test_float64_target_precision_is_retained():
    class Copy(torch.nn.Module):
        def forward(self,x):
            return x[:,:1]
    data,norm=fixture()
    data['fields']=np.ones_like(data['fields'])
    data['fields'][:,1:]+=1e-9
    result,_=evaluate_archive(Copy(),norm,data,'precision')
    # If targets were cast to float32 before diagnostics this would disappear.
    assert result['summary']['teacher_forced']['relative_l2']['mean'][1] > 0
