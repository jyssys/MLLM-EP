"""Pure selection rules for request-scoped checkpoint exposure."""
from dataclasses import dataclass
from ..completion_aware_loop_pd.completion import plan_completion, suffix_execution_cost


@dataclass(frozen=True)
class CheckpointChoice:
    target: int
    reason: str
    predicted_D_ms: float | None = None


def choose_checkpoint(depth, total, checkpoints, advertisement, now, config, pressure):
    """Choose a legal future boundary. This does not commit ownership."""
    if not advertisement:
        return CheckpointChoice(total, 'NO_ADVERTISEMENT')
    age_ms = (now - advertisement['observed']) * 1000
    if age_ms < 0 or age_ms > config.get('advertisement_max_age_ms', 50.):
        return CheckpointChoice(total, 'STALE_ADVERTISEMENT')
    if not advertisement.get('available'):
        return CheckpointChoice(total, 'DESTINATION_UNAVAILABLE')
    if pressure < config.get('minimum_P_pressure', 2) and not config.get('qualification_allow_unpressured'):
        return CheckpointChoice(total, 'NO_P_PRESSURE')
    future = advertisement.get('future_quantum_budget_ms', advertisement.get('budget_ms', 0.))
    if future < config.get('minimum_quantum_ms', 1.):
        return CheckpointChoice(total, 'NO_EXECUTION_BUDGET')
    for cut in sorted(checkpoints):
        if not depth < cut < total:
            continue
        ceiling = advertisement.get('recovery_remaining_ceiling')
        if ceiling is not None and total-cut > ceiling:
            continue
        work, floor = suffix_execution_cost(config['execution_candidates'], total-cut,
            config['length'], future, config.get('guard_ms', 0.), False)
        plan = plan_completion(work, floor, advertisement.get('budget_ms', 0.), future,
            advertisement.get('decode_reserve_ms', 0.), config.get('handoff_upper_ms', 100.),
            0., config.get('checkpoint_selection_deadline_ms', config.get('completion_deadline_ms', 1200.)))
        if plan.feasible:
            return CheckpointChoice(cut, 'FORECAST_FEASIBLE_REVALIDATE_AT_BOUNDARY', plan.predicted_ms)
    return CheckpointChoice(total, 'NO_COMPLETION_FEASIBLE_CHECKPOINT')
