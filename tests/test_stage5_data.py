import numpy as np
import pytest
torch=pytest.importorskip('torch')
from neural_operator.normalization import Normalization
from neural_operator.data import trajectory_split
from stage5_training.data import RolloutWindowDataset


def fixture():
    data={'fields':np.arange(64*11*4*4,dtype=float).reshape(64,11,4,4),
          'coefficients':np.arange(64*3,dtype=float).reshape(64,3),'times':np.linspace(0,1,11)}
    return data,Normalization(2,3,[1,2,3],[2,3,4])


def test_window_shapes_alignment_and_split_membership(monkeypatch):
    data,norm=fixture();splits=trajectory_split()
    monkeypatch.setattr(Normalization,'fit',lambda *args:pytest.fail('Never refit normalization'))
    for name,ids in splits.items():
        dataset=RolloutWindowDataset(data,ids,norm)
        assert len(dataset)==6*len(ids)
        assert set(i for i,k in dataset.windows)==set(ids)
        for i in ids:
            assert [k for j,k in dataset.windows if j==i]==list(range(6))
        x,y,m0,i,k=dataset[5]
        assert x.shape==(4,4,4) and y.shape==(5,1,4,4) and k==5
        np.testing.assert_allclose(y[:,0],norm.field(data['fields'][i,6:11]),rtol=1e-6)
        np.testing.assert_allclose(norm.inverse_field(x[0].double()),data['fields'][i,5],rtol=1e-6)
        assert m0.item()==data['fields'][i,5].sum()*(2*np.pi/4)**2
        np.testing.assert_allclose(x[1:,0,0],norm.coefficients(data['coefficients'][i]),rtol=1e-6)
    assert set(splits['train']).isdisjoint(splits['validation'])
    assert set(splits['train']).isdisjoint(splits['test'])


@pytest.mark.parametrize('steps',[0,11,True,2.5])
def test_invalid_window_length(steps):
    data,norm=fixture()
    with pytest.raises(ValueError):RolloutWindowDataset(data,[0],norm,steps)
