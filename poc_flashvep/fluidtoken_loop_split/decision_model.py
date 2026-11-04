"""CPU-only pressure and cost calculations. No uncalibrated runtime fallback."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class QueuedRequest:
    arrival_s: float
    predicted_prefix_s: float
    ttft_limit_s: float


def prefill_pressure(requests, now_s):
    cumulative, pressure = 0., 0.
    for request in requests:
        if request.predicted_prefix_s < 0 or request.ttft_limit_s <= 0:
            raise ValueError('Invalid service prediction or TTFT limit')
        cumulative += request.predicted_prefix_s
        remaining = request.arrival_s + request.ttft_limit_s - now_s
        pressure = max(pressure, cumulative / max(remaining, 1e-9))
    return pressure


def decode_pressure(active_blocks, reserved_blocks, total_blocks,
                    active_requests, safe_batch):
    if not (0 <= active_blocks <= total_blocks and 0 <= reserved_blocks <= total_blocks - active_blocks):
        raise ValueError('Active and reserved KV must be disjoint legal counts')
    if total_blocks <= 0 or safe_batch <= 0 or active_requests < 0:
        raise ValueError('Invalid Decode capacity')
    return dict(load=max(active_blocks / total_blocks, active_requests / safe_batch),
                admission=(active_blocks + reserved_blocks) / total_blocks,
                free_blocks=total_blocks - active_blocks - reserved_blocks)


@dataclass(frozen=True)
class SplitCost:
    # Values must come from explicit calibration. No default latency exists.
    key: int
    p_service_s: float
    d_continuation_s: float
    transfer_exposed_s: float
    d_extra_blocks: int


def choose(candidates, full, p_queue_service_s, d_queue_service_s,
           d_iteration_s, token_age_s, itl_limit_s, tpot_margin_s,
           free_blocks, pressure, has_continuation=False):
    """Select only legal observed-state actions with measured cost predictions.

    This shared cost selector is not the complete FluidToken queue algorithm.
    A runtime adapter must expose that limitation and recheck at handoff.
    """
    if full not in candidates:
        raise ValueError('A Full-PD fallback is mandatory')
    if pressure <= 1 or has_continuation:
        return full, 'FULL_PD_NO_PRESSURE_OR_EXISTING_CONTINUATION'
    slack = min(itl_limit_s - token_age_s, tpot_margin_s) - d_iteration_s
    feasible = []
    for candidate in candidates:
        numbers = (candidate.p_service_s, candidate.d_continuation_s,
                   candidate.transfer_exposed_s)
        if not all(isfinite(value) and value >= 0 for value in numbers):
            raise ValueError('Missing or invalid calibrated cost')
        if candidate.d_extra_blocks < 0:
            raise ValueError('Invalid workspace or KV requirement')
        if candidate != full and (candidate.d_continuation_s > slack
                or candidate.d_extra_blocks > free_blocks):
            continue
        completion = (p_queue_service_s + candidate.p_service_s
                      + candidate.transfer_exposed_s)
        if candidate.d_continuation_s:
            completion = max(completion, d_queue_service_s) + candidate.d_continuation_s
        feasible.append((completion, candidate.key, candidate))
    result = min(feasible, key=lambda row: row[:2])[2]
    return result, 'FULL_PD_COST_OR_SLACK' if result == full else 'CALIBRATED_SLACK_ACCEPT'


def queue_relief_choice(candidates, full, later_deficit_s, d_iteration_s,
                        token_age_s, itl_limit_s, tpot_margin_s, free_blocks,
                        pressure, has_continuation=False):
    """Allocate bounded relief to an earlier request for known queued deadlines.

    This discrete-cost adaptation follows FluidToken's queue-relief principle.
    It is not the paper's complete continuous-token optimizer.
    """
    if full not in candidates:
        raise ValueError('A Full-PD fallback is mandatory')
    if not isfinite(later_deficit_s) or later_deficit_s < 0:
        raise ValueError('Invalid observed queue deficit')
    if pressure <= 1 or has_continuation or later_deficit_s == 0:
        return full, 'FULL_PD_NO_LATER_DEFICIT_OR_PENDING_WORK'
    slack = min(itl_limit_s - token_age_s, tpot_margin_s) - d_iteration_s
    feasible = []
    for candidate in candidates:
        if not all(isfinite(v) and v >= 0 for v in
                   (candidate.p_service_s, candidate.d_continuation_s, candidate.transfer_exposed_s)):
            raise ValueError('Missing or invalid calibrated cost')
        if candidate.d_extra_blocks < 0:
            raise ValueError('Invalid capacity requirement')
        if candidate == full or candidate.d_continuation_s > slack or candidate.d_extra_blocks > free_blocks:
            continue
        relief = (full.p_service_s - candidate.p_service_s
                  - max(0., candidate.transfer_exposed_s - full.transfer_exposed_s))
        if relief > 0:
            feasible.append((relief, candidate.d_continuation_s, candidate.key, candidate))
    if not feasible:
        return full, 'FULL_PD_NO_SAFE_QUEUE_RELIEF'
    sufficient = [row for row in feasible if row[0] >= later_deficit_s]
    if sufficient:
        selected = min(sufficient, key=lambda row: row[:3])[-1]
    else:
        selected = min(feasible, key=lambda row: (-row[0], row[1], row[2]))[-1]
    return selected, 'EARLIEST_QUEUE_BUDGET_RELIEF'
