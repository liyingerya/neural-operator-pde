import math
import pytest
pytest.importorskip('torch')
from stage4_evaluation.benchmark import measure


def test_finite_positive_timings():
    result=measure(lambda:sum(range(100)),repeats=3,warmups=1,min_group_seconds=0.001)
    assert len(result['samples_seconds'])==3
    assert all(math.isfinite(t) and t>0 for t in result['samples_seconds'])
    assert result['q25_seconds']<=result['median_seconds']<=result['q75_seconds']


def test_bad_timing_configuration():
    with pytest.raises(ValueError):
        measure(lambda:None,repeats=0)
