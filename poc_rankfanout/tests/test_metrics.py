import math

from poc_rankfanout.rankfanout.metrics import improvement_pct, summarize_latency


def test_summary():
    out = summarize_latency([1, 2, 3, 4])
    assert out["n"] == 4
    assert out["median"] == 2.5
    assert out["iqr"] == 1.5


def test_empty_summary():
    assert math.isnan(summarize_latency([])["median"])


def test_improvement():
    assert improvement_pct(10, 8) == 20
