import json
from pathlib import Path
import pytest
pytest.importorskip('torch')
from stage6_control.reporting import budget_curves
from stage4_evaluation.checkpoint import sha256


def test_saved_budget_histories_are_read_without_mutation(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('runs/stage5').mkdir(parents=True)
    Path('runs/stage5/selection.json').write_text(json.dumps({'B':{'directory':'rollout'},'C':{'directory':'selected_mass'}}))
    old=[]
    for folder in ['rollout','selected_mass']:
        path=Path('runs/stage5')/folder/'history.json';path.parent.mkdir()
        path.write_text(json.dumps([{'epoch':i,'runtime_seconds':2.,
            'validation':{'final_relative_l2':1/i},'training':{'field':0.1}} for i in range(1,21)]))
        old.append((path,sha256(path)))
    run=Path('runs/stage6');run.mkdir()
    (run/'history.json').write_text(json.dumps([{'epoch':i,'optimizer_updates':60*i,'supervised_fields':480*i,
        'cumulative_epoch_seconds':i*1.5,'cumulative_training_seconds':float(i),'training_loss':0.1,
        'validation':{'final_relative_l2':1/i,'one_step_relative_l2':0.1}} for i in range(1,61)]))
    curves,comparisons,hashes=budget_curves(run)
    assert len(comparisons['supervised_fields'])==20
    assert comparisons['supervised_fields'][-1]['supervised_fields']==28800
    assert comparisons['optimizer_updates'][-1]['optimizer_updates']==720
    assert curves['B'][-1]['cumulative_epoch_seconds']==40
    assert curves['D'][-1]['optimizer_updates']==3600
    assert curves['C'][-1]['supervised_fields']==28800
    assert all(sha256(p)==h==hashes[str(p)] for p,h in old)
