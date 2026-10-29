"""Final-review accounting tests. They never create a GPU context."""
import pytest

from .finalize import PHASES, interval, phase_vector, strongest


def test_strongest_selects_the_requested_measured_metric():
    rows = [{'name': 'A', 'latency': '12', 'TTFT': '1'},
            {'name': 'B', 'latency': '10', 'TTFT': '3'}]
    assert strongest(rows, 'latency')['name'] == 'B'
    assert strongest(rows, 'TTFT')['name'] == 'A'


def test_partition_uses_summed_request_time_not_mean_of_percentages():
    rows = [dict(phase=PHASES[0], N='10', phase_mean_ms='1'),
            dict(phase=PHASES[1], N='10', phase_mean_ms='9'),
            dict(phase=PHASES[0], N='1', phase_mean_ms='100')]
    result = phase_vector(rows)
    assert sum(result.values()) == pytest.approx(100)
    assert result[PHASES[0]] == pytest.approx(55)
    assert result[PHASES[1]] == pytest.approx(45)


def test_interval_preserves_missing_or_unbounded_uncertainty():
    assert interval('UNBOUNDED_OR_UNDEFINED_ZERO_BASELINE_DRAWS') == 'UNBOUNDED_OR_UNDEFINED_ZERO_BASELINE_DRAWS'
    assert interval('[0.95, 1.20]') == '0.950–1.200×'


def test_empty_partition_is_not_filled_with_zero_measurements():
    with pytest.raises(ValueError, match='denominator'):
        phase_vector([])
