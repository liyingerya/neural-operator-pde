from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('torch')
from dataset_generation import generate_dataset
from stage4_evaluation.ood import configurations,verify_matched,internal_steps,numerical_checks


@pytest.mark.parametrize('name',['fresh_id','high_nu','velocity_pp','velocity_pn','velocity_np','velocity_nn'])
def test_distributions_and_determinism(name):
    config=replace(configurations()[name],num_trajectories=2,grid_size=16,num_snapshots=3,t_final=0.2)
    a=generate_dataset(config);b=generate_dataset(config)
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
    c=a['coefficients']
    if name=='high_nu':
        assert np.all((c[:,2]>=0.055)&(c[:,2]<=0.08)) and np.all(c[:,2]>0.05)
    if name.startswith('velocity'):
        assert np.all((np.abs(c[:,:2])>=1.05)&(np.abs(c[:,:2])<=1.4))
        for j,sign in enumerate(name.split('_')[1]):
            assert np.all(c[:,j]>1) if sign=='p' else np.all(c[:,j]<-1)
    assert numerical_checks(a)['finite']


def test_matching_resolution_and_mismatch_detection():
    config=replace(configurations(True)['resolution_64'],num_trajectories=2,grid_size=16,num_snapshots=2,t_final=0.1)
    coarse=generate_dataset(config);fine=generate_dataset(replace(config,grid_size=32))
    verify_matched(coarse,fine)
    fine['coefficients'][0,0]+=0.1
    with pytest.raises(ValueError,match='coefficients'):
        verify_matched(coarse,fine)


def test_step_count():
    assert internal_steps(0.1,0.03)==4
    assert internal_steps(0.1,0.2)==1
