import numpy as np
import pytest
torch=pytest.importorskip('torch')
from stage5_training.config import TrainingConfig
from stage5_training.training import fresh_model,state_hash,gate_passed,selection_key,loader
from stage5_training.data import RolloutWindowDataset
from neural_operator.normalization import Normalization
from stage4_evaluation.rollout import predict_trajectories


def test_deterministic_initialization_and_loader():
    config=TrainingConfig();architecture=dict(modes=2,width=4,blocks=1,projection_width=8)
    first=fresh_model(config,architecture);second=fresh_model(config,architecture)
    assert state_hash(first)==state_hash(second)
    data={'fields':np.ones((2,11,4,4)),'coefficients':np.ones((2,3)),'times':np.linspace(0,1,11)}
    dataset=RolloutWindowDataset(data,[0,1],Normalization(0,1,[0,0,0],[1,1,1]))
    def order():return [(i.tolist(),k.tolist()) for _,_,_,i,k in loader(dataset,config,True)]
    assert order()==order()


def test_hard_gate_checks_mass_for_C():
    initial={'field':1.,'mass':1.}
    final={'field':0.01,'mass':0.2,'physical_relative_l2':0.02}
    assert gate_passed(initial,final,0)
    assert not gate_passed(initial,final,0.1)
    final['mass']=0.01
    assert gate_passed(initial,final,0.1)
    final['field']=0.5
    assert not gate_passed(initial,final,0.1)


def test_selection_prioritizes_validation_field_error():
    a={'final_relative_l2':0.1,'final_mass_error':0.5}
    b={'final_relative_l2':0.2,'final_mass_error':0.001}
    assert selection_key(a,5)<selection_key(b,1)


def test_unchanged_stage4_rollout_accepts_new_weights():
    model=fresh_model(TrainingConfig(),dict(modes=2,width=4,blocks=1,projection_width=8))
    fields=np.ones((1,3,8,8));coeff=np.array([[0.1,0.2,0.03]])
    norm=Normalization(0.1,0.2,[0,0,0],[1,1,1])
    predictions,failures=predict_trajectories(model,norm,fields,coeff,np.array([0,0.1,0.2]))
    assert predictions['autoregressive'].shape==fields.shape
    assert not any(failures.values())


def test_failed_gate_prevents_all_full_runs(tmp_path,monkeypatch):
    import json
    from examples import train_stage5
    from neural_operator.fno import FNO2d
    model=FNO2d(modes=2,width=4,blocks=1,projection_width=8)
    norm=Normalization(0,1,[0,0,0],[1,1,1])
    split={'train':list(range(48)),'validation':list(range(48,56)),'test':list(range(56,64))}
    baseline={'split_ids':split,'archive_sha256':'fixture'}
    data={'fields':np.ones((64,11,4,4)),'coefficients':np.ones((64,3)),'times':np.linspace(0,1,11)}
    monkeypatch.setattr(train_stage5,'load_frozen',lambda:(model,norm,baseline))
    monkeypatch.setattr(train_stage5,'load_dataset',lambda path:data)
    monkeypatch.setattr(train_stage5,'tiny_gate',lambda *args:{'passed':False})
    monkeypatch.setattr(train_stage5,'run_training',lambda *args:pytest.fail('Full training after failed gate'))
    output=tmp_path/'blocked'
    monkeypatch.setattr('sys.argv',['train_stage5','--output',str(output)])
    with pytest.raises(SystemExit,match='gate failed'):
        train_stage5.main()
    assert json.loads((output/'stopped.json').read_text())['reason']=='tiny_overfit_gate_failed'
    assert not (output/'rollout').exists()
