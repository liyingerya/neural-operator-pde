import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from neural_operator.data import trajectory_split
from neural_operator.normalization import Normalization
from stage5_training.training import fresh_model,state_hash,selection_key
from stage6_control.config import Config,budget,classify
from stage6_control.provenance import pairs,verify_snapshot,frozen_context,load_d
from stage6_control.training import validation_metrics
from stage4_evaluation.checkpoint import sha256


ARCH={'modes':2,'width':4,'blocks':1,'projection_width':8}


def test_frozen_pairs_normalization_and_no_leakage(monkeypatch):
    data={'fields':np.ones((64,11,64,64)),'coefficients':np.ones((64,3)),
          'times':np.linspace(0,1,11)}
    norm=Normalization(0.2,0.5,[0,0,0],[1,2,3]);split=trajectory_split()
    monkeypatch.setattr(Normalization,'fit',lambda *args:pytest.fail('Must never fit'))
    datasets=pairs(data,split,norm)
    assert [len(datasets[k]) for k in ['train','validation','test']]==[480,80,80]
    for key,d in datasets.items():
        assert d.normalization is norm
        assert {i for i,k in d.pairs}==set(split[key])
        assert d.pairs==[(i,k) for i in split[key] for k in range(10)]
        x,y,i,k=d[0]
        assert x.shape==(4,64,64) and y.shape==(1,64,64)
        np.testing.assert_allclose(norm.inverse_field(x[0]),data['fields'][i,k])
    split['test'][0]=split['train'][0]
    with pytest.raises(ValueError,match='leakage'):pairs(data,split,norm)


def test_frozen_context_exact_reuse_and_initialization(tmp_path,monkeypatch):
    from stage6_control import provenance
    cfg=Config();model=fresh_model(cfg,ARCH);norm=Normalization(0,1,[0,0,0],[1,1,1])
    previous={'architecture':ARCH,'normalization':norm.to_dict(),'split_ids':trajectory_split(),
              'initial_state_sha256':state_hash(model),'source_sha256':{}}
    monkeypatch.chdir(tmp_path);Path('runs/stage5').mkdir(parents=True)
    path=Path('runs/stage5/protocol.json');path.write_text(json.dumps(previous))
    monkeypatch.setattr(provenance,'load_frozen',lambda:(model,norm,{'split_ids':previous['split_ids']}))
    loaded,_,protocol,control=frozen_context(cfg)
    assert loaded is norm and protocol==previous
    assert state_hash(control)==state_hash(fresh_model(cfg,ARCH))
    previous['normalization']['field_std']=2
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError,match='normalization'):frozen_context(cfg)


def test_budget_matches_targets_not_updates():
    assert budget(60)=={'epoch':60,'optimizer_updates':3600,'supervised_fields':28800}
    assert budget(20,288,5)=={'epoch':20,'optimizer_updates':720,'supervised_fields':28800}
    assert budget(0)['supervised_fields']==0


def test_selection_uses_rollout_not_one_step():
    better={'final_relative_l2':0.1,'final_mass_error':0.4,'one_step_relative_l2':0.2}
    worse={'final_relative_l2':0.2,'final_mass_error':0.01,'one_step_relative_l2':0.01}
    assert selection_key(better,20)<selection_key(worse,1)
    assert selection_key(better,2)<selection_key(better,3)


def test_validation_compatible_with_frozen_stage5():
    from stage5_training.training import validation_metrics as previous_validation
    model=fresh_model(Config(),ARCH);norm=Normalization(0,1,[0,0,0],[1,1,1])
    data={'fields':np.ones((3,11,8,8)),'coefficients':np.ones((3,3)),
          'times':np.linspace(0,1,11)}
    new=validation_metrics(model,data,[1],norm)
    old=previous_validation(model,data,[1],norm)
    for key in ['final_relative_l2','final_mass_error','relative_l2_by_horizon','mass_error_by_horizon']:
        assert new[key]==old[key]
    assert np.isfinite(new['one_step_relative_l2'])


def test_protected_file_mutation_rejected(tmp_path):
    path=tmp_path/'protected';path.write_text('original')
    snapshot={str(path):sha256(path)}
    assert verify_snapshot(snapshot)==1
    path.write_text('modified')
    with pytest.raises(ValueError,match='hash mismatch'):verify_snapshot(snapshot)


@pytest.mark.parametrize('values,expected',[
    ((0.17,0.29,0.28,0.18),'case_1'),((0.17,0.29,0.28,0.285),'case_2'),
    ((0.17,0.29,0.28,0.24),'case_3'),((0.17,0.29,0.28,0.5),'outside_predeclared_cases')])
def test_predeclared_cases(values,expected):
    assert classify(*values)==expected


def test_selected_d_provenance(tmp_path):
    model=fresh_model(Config(),ARCH)
    protocol={'architecture':ARCH,'normalization':Normalization(0,1,[0,0,0],[1,1,1]).to_dict(),
              'split_ids':trajectory_split(),'config':Config().to_dict(),
              'initial_state_sha256':state_hash(model),'source_sha256':{}}
    state={k:v for k,v in protocol.items() if k!='source_sha256'}
    state.update(model_state_dict=model.state_dict(),epoch=3,validation={'final_relative_l2':0.1})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol));torch.save(state,tmp_path/'best.pt')
    selection={'gate_passed':True,'test_or_ood_used':False,'epoch':3,'validation':state['validation'],
               'protocol_sha256':sha256(tmp_path/'protocol.json'),'sha256':sha256(tmp_path/'best.pt')}
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection))
    loaded,norm,_=load_d(tmp_path)
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert norm.to_dict()==protocol['normalization']
    selection['test_or_ood_used']=True;path.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='validation-only'):load_d(tmp_path)


def test_failed_gate_blocks_full_training(tmp_path,monkeypatch):
    from examples import train_stage6
    model=fresh_model(Config(),ARCH);norm=Normalization(0,1,[0,0,0],[1,1,1])
    previous={'architecture':ARCH,'split_ids':trajectory_split()}
    monkeypatch.chdir(tmp_path)
    Path('runs/stage5').mkdir(parents=True);Path('runs/stage5/protocol.json').write_text('{}')
    Path('examples').mkdir();Path('examples/train_stage6.py').write_text('# fixture')
    monkeypatch.setattr(train_stage6,'frozen_context',lambda cfg:(norm,{'archive_sha256':'fixture'},previous,model))
    monkeypatch.setattr(train_stage6,'protected_snapshot',lambda:{})
    original_check_output=train_stage6.subprocess.check_output
    monkeypatch.setattr(train_stage6.subprocess,'check_output',
        lambda command,**kwargs:'fixture-revision' if command[0]=='git' else original_check_output(command,**kwargs))
    monkeypatch.setattr(train_stage6,'load_dataset',lambda path:{})
    monkeypatch.setattr(train_stage6,'pairs',lambda *args:{'train':list(range(480))})
    monkeypatch.setattr(train_stage6,'tiny_gate',lambda *args:{'passed':False})
    monkeypatch.setattr(train_stage6,'full_training',lambda *args:pytest.fail('Training after failed gate'))
    monkeypatch.setattr('sys.argv',['train_stage6'])
    with pytest.raises(SystemExit,match='gate failed'):train_stage6.main()
    assert Path('runs/stage6/stopped.json').exists()
    assert not Path('runs/stage6/selection.json').exists()
