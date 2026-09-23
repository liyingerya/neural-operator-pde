import json
import pytest
torch=pytest.importorskip('torch')
from neural_operator.fno import FNO2d
from neural_operator.normalization import Normalization
from stage4_evaluation import checkpoint


def test_verified_frozen_loading_and_tamper_rejection(tmp_path,monkeypatch):
    model=FNO2d(modes=2,width=4,blocks=1,projection_width=8)
    normalization=Normalization(0,1,[0,0,0],[1,1,1]).to_dict()
    state={'epoch':31,'architecture':model.config,'normalization':normalization,
           'split_ids':{'test':[1]},'model_state_dict':model.state_dict()}
    torch.save(state,tmp_path/'best.pt')
    (tmp_path/'archive.npz').write_bytes(b'archive')
    (tmp_path/'source.py').write_text('source')
    digest=checkpoint.sha256(tmp_path/'best.pt')
    manifest={'selected_checkpoint':{'epoch':31,'path':'best.pt','sha256':digest},
              'archive_sha256':checkpoint.sha256(tmp_path/'archive.npz'),
              'source_sha256':{'source.py':checkpoint.sha256(tmp_path/'source.py')},
              'normalization':normalization,'split_ids':state['split_ids'],'architecture':model.config}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(checkpoint,'APPROVED_HASH',digest)
    monkeypatch.setattr(checkpoint,'ROOT',tmp_path)
    monkeypatch.setattr(Normalization,'fit',lambda *args:pytest.fail('Never refit'))
    loaded,norm,_=checkpoint.load_frozen(tmp_path,tmp_path/'archive.npz')
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert norm.to_dict()==normalization
    (tmp_path/'source.py').write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        checkpoint.load_frozen(tmp_path,tmp_path/'archive.npz')


def test_exclusive_json_output(tmp_path):
    path=tmp_path/'result.json'
    checkpoint.write_json(path,{'value':1})
    with pytest.raises(FileExistsError):
        checkpoint.write_json(path,{'value':2})
