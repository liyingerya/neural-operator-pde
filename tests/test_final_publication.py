"""Lightweight consistency checks for the frozen public synthesis."""
import csv
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
from examples.build_final_report import build_summary,FIGURES

ROOT=Path(__file__).resolve().parents[1]


def test_final_summary_and_csv_match_frozen_sources():
    expected=build_summary(ROOT)
    actual=json.loads((ROOT/'docs/results/final_summary.json').read_text())
    assert actual==expected
    rows=list(csv.DictReader((ROOT/'docs/results/final_model_comparison.csv').open()))
    assert [r['model'] for r in rows]==['A','B','C','D']
    for row in rows:
        for key,value in expected['models'][row['model']].items():
            assert (float(row[key])==value) if isinstance(value,(int,float)) else (row[key]==value)
    assert expected['persistence']['one_step_id_l2_published_rounded']!=expected['persistence']['final_id_rollout_l2']


def markdown_links(path):
    text=path.read_text()
    # Code examples are not links; allow descriptive labels and optional fragments.
    text=re.sub(r'```.*?```','',text,flags=re.S)
    return re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)


def test_public_documentation_links_resolve():
    paths=[ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md'))]
    for path in paths:
        if path.name in ['portfolio_notes.md','stage1_solver_walkthrough.md','stage2_learning_notes.md']:
            continue
        for target in markdown_links(path):
            if re.match(r'^[a-zA-Z]+:',target) or target.startswith('#'):continue
            destination=unquote(target.split('#',1)[0].strip('<>'))
            assert (path.parent/destination).exists(),(path,target)


def test_only_curated_small_figures_are_public():
    figures=list((ROOT/'docs/figures').glob('*.png'))
    assert {p.stem for p in figures}==set(FIGURES)
    for path in figures:
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert path.stat().st_size<500_000


def test_no_prohibited_tracked_artifacts():
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    forbidden={'.npz','.npy','.pt','.pth','.pkl','.pyc','.log'}
    for name in filter(None,files):
        p=Path(name)
        assert p.parts[0] not in {'runs','data','.venv','__pycache__'}
        assert p.suffix not in forbidden
        assert p.name not in {'portfolio_notes.md','stage1_solver_walkthrough.md','stage2_learning_notes.md'}
        if p.suffix=='.png':assert p.parent==Path('docs/figures') and p.stem in FIGURES
