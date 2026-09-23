"""Trajectory grouping and training-only transforms."""
import numpy as np
import pytest
pytest.importorskip('torch')
from neural_operator.data import trajectory_split, TrajectoryPairDataset, validate_baseline
from neural_operator.normalization import Normalization
from neural_operator.training import make_loader


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    return {'fields': rng.normal(size=(64,11,8,8)),
            'coefficients': rng.normal(size=(64,3)), 'times': np.linspace(0,1,11)}


def test_split_and_pair_membership(data):
    split = trajectory_split()
    assert split == trajectory_split()
    assert split != trajectory_split(seed=2028)
    assert [len(x) for x in split.values()] == [48,8,8]
    all_ids = sum(split.values(), [])
    assert sorted(all_ids) == list(range(64))
    norm = Normalization.fit(data, split['train'])
    for name, ids in split.items():
        dataset = TrajectoryPairDataset(data, ids, norm)
        assert len(dataset) == 10*len(ids)
        assert set(i for i,k in dataset.pairs) == set(ids)
        for i in ids:
            assert [k for j,k in dataset.pairs if j==i] == list(range(10))


def test_pair_alignment_broadcast_and_batch(data):
    norm = Normalization.fit(data, [2,5,9])
    dataset = TrajectoryPairDataset(data, [5,2], norm)
    x,y,i,k = dataset[12]
    assert (i,k) == (2,2)
    assert x.shape == (4,8,8) and y.shape == (1,8,8)
    np.testing.assert_allclose(x[0], norm.field(data['fields'][2,2]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y[0], norm.field(data['fields'][2,3]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1:,0,0], norm.coefficients(data['coefficients'][2]), rtol=1e-6)
    assert (x[1:] == x[1:,:1,:1]).all()
    batch = next(iter(make_loader(dataset)))
    assert batch[0].shape == (8,4,8,8) and batch[1].shape == (8,1,8,8)


def test_loader_shuffle_reproducible(data):
    dataset = TrajectoryPairDataset(data, [1,3], Normalization.fit(data,[1,3]))
    def order():
        return [(i.tolist(),k.tolist()) for _,_,i,k in make_loader(dataset,shuffle=True)]
    assert order() == order()


def test_baseline_schedule(data):
    data['fields'] = np.broadcast_to(data['fields'][:,:,:1,:1], (64,11,64,64))
    validate_baseline(data)
    data['times'][-1] += 0.01
    with pytest.raises(ValueError):
        validate_baseline(data)


@pytest.mark.parametrize('ids', [[], [1,1], [-1], [64]])
def test_bad_pair_ids(data, ids):
    with pytest.raises(ValueError):
        TrajectoryPairDataset(data, ids, Normalization.fit(data,[1]))
