import numpy as np
import pytest
torch=pytest.importorskip('torch')
from neural_operator.data import TrajectoryPairDataset
from neural_operator.normalization import Normalization
from neural_operator.fno import FNO2d
from neural_operator.training import seed_everything,make_loader,train_epoch,evaluate,overfit_passed


def test_gate_requires_both_conditions():
    initial={'normalized_mse':1.0}
    assert overfit_passed(initial,{'normalized_mse':0.01,'relative_l2':0.04})
    assert not overfit_passed(initial,{'normalized_mse':0.1,'relative_l2':0.04})
    assert not overfit_passed(initial,{'normalized_mse':0.01,'relative_l2':0.1})
    assert not overfit_passed(initial,{'normalized_mse':float('nan'),'relative_l2':0.01})


def test_train_evaluate_and_checkpoint(tmp_path):
    seed_everything()
    rng=np.random.default_rng(3)
    data={'fields':rng.uniform(size=(2,3,8,8)), 'coefficients':rng.normal(size=(2,3))}
    norm=Normalization.fit(data,[0,1])
    dataset=TrajectoryPairDataset(data,[0,1],norm)
    loader=make_loader(dataset)
    model=FNO2d(width=4,modes=2,blocks=1,projection_width=8)
    before=model.lift.weight.detach().clone()
    loss=train_epoch(model,loader,torch.optim.Adam(model.parameters(),lr=1e-3))
    assert np.isfinite(loss) and not torch.equal(before,model.lift.weight)
    result=evaluate(model,loader,norm)
    assert len(result['pairs'])==4 and np.isfinite(result['relative_mass_error'])
    persistence=evaluate(model,loader,norm,persistence=True)
    assert np.isfinite(persistence['relative_l2'])
    path=tmp_path/'model.pt'
    torch.save(model.state_dict(),path)
    restored=FNO2d(width=4,modes=2,blocks=1,projection_width=8)
    restored.load_state_dict(torch.load(path,weights_only=True))
    assert evaluate(restored,loader,norm) == result


def test_failed_gate_blocks_full_training(tmp_path, monkeypatch):
    import json
    from examples import train_fno
    archive=tmp_path/'archive.npz'
    archive.write_bytes(b'fixture')
    data={'fields':np.zeros((64,11,1,1)), 'coefficients':np.ones((64,3))}
    data['fields']=np.broadcast_to(data['fields'],(64,11,64,64))
    data['times']=np.linspace(0,1,11)
    monkeypatch.setattr(train_fno,'load_dataset',lambda path:data)
    monkeypatch.setattr(train_fno,'overfit_gate',lambda *args: {
        'passed':False,'epochs':300,'criteria':{},'runtime_seconds':0})
    def forbidden(*args,**kwargs):
        pytest.fail('Full training must not start after failed gate')
    monkeypatch.setattr(train_fno,'train_epoch',forbidden)
    output=tmp_path/'run'
    monkeypatch.setattr('sys.argv',['train_fno','--archive',str(archive),'--output',str(output)])
    with pytest.raises(SystemExit,match='Overfit gate failed'):
        train_fno.main()
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['status']=='stopped_overfit_gate_failed'
    assert not (output/'best.pt').exists()
