import numpy as np
import pytest

from poc_rankfanout.rankfanout.routing import balanced_fanout_route, validate_control_family


@pytest.mark.parametrize("num_tokens", [128, 256, 512, 1024, 2048, 4096, 8192])
def test_histogram_identical_across_fanout(num_tokens):
    routes = [balanced_fanout_route(num_tokens, fanout, seed=11) for fanout in range(1, 5)]
    result = validate_control_family(routes)
    assert result["rank_load"] == [2 * num_tokens] * 4
    assert result["expert_load"] == [num_tokens // 16] * 128
    assert np.asarray(routes).shape == (4, num_tokens, 8)


def test_requires_exact_divisibility():
    with pytest.raises(ValueError):
        balanced_fanout_route(127, 1)
