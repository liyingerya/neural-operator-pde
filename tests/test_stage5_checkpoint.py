import json
import pytest
torch=pytest.importorskip('torch')
from neural_operator.fno import FNO2d
from stage4_evaluation.checkpoint import sha256,APPROVED_HASH
from stage5_training.checkpoint import load_candidate


def test_candidate_provenance_and_frozen_selection(tmp_path):
    model=FNO2d(modes=2,width=4,blocks=1,projection_width=8)
    norm={'field_mean':0.,'field_std':1.,'coefficient_mean':[0,0,0],'coefficient_std':[1,1,1]}
    protocol={'architecture':model.config,'normalization':norm,'split_ids':{'train':[0],'validation':[1],'test':[2]},
              'config':{'seed':2028},'initial_state_sha256':'initial','historical_checkpoint_sha256':APPROVED_HASH,'source_sha256':{}}
    state={k:protocol[k] for k in ['architecture','normalization','split_ids','config','initial_state_sha256']}
    state.update(model_state_dict=model.state_dict(),epoch=2,lambda_mass=0.,validation={'final_relative_l2':0.1})
    (tmp_path/'rollout').mkdir();path=tmp_path/'rollout/best.pt';torch.save(state,path)
    selection={'all_gates_passed':True,'test_or_ood_used':False,'B':{'directory':'rollout','checkpoint':'best.pt',
               'sha256':sha256(path),'epoch':2,'validation':state['validation']}}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol));(tmp_path/'selection.json').write_text(json.dumps(selection))
    loaded,normalization,_=load_candidate(tmp_path,'B')
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert normalization.to_dict()==norm
    selection['test_or_ood_used']=True
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='separation'):load_candidate(tmp_path,'B')


def test_no_evaluation_before_selection(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(FileNotFoundError):load_candidate(tmp_path,'B')
