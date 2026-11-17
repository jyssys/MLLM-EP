"""Retest host-control optimization and tune the token destination fairly."""
import fcntl
import json
from .common import ROOT, QUIET_LOCK, resources, write
from .axis_configuration import base
from .discovery import job
from .token_axis import matched_token_plan


def main():
    assert json.loads((ROOT / 'analysis/fastlane_Layer_all_cuts_state_N32_v1_full_state_gate.json').read_text())['passed']
    configs = []
    for repeat, enabled in enumerate((False, True, False)):
        configs.append(base(f'credit_ABA_{repeat}_fast{int(enabled)}_Last12_P4_D4_N128_q6.4_v1',
            method='Static', count=128, rate=6.4, layers=12, credit_fast_lane=enabled,
            hidden_source_slots=32, hidden_destination_slots=32,
            D_combine_ready_macros=True, D_wait_for_source_macro=False))
    for destination in (4, 8):
        for suffix in (2048, 2672, 4096):
            configs.append(base(f'fair_Token_S{suffix}_P4_D{destination}_N128_q6.4_v1',
                method='Token', count=128, rate=6.4, suffix=suffix,
                D_admission_batch=destination, D_combine_ready_macros=True,
                D_wait_for_source_macro=False))
    for layers in (8, 12):
        for method in ('Static', 'Token'):
            cfg = base(f'matched_rows_{method}_Last{layers}_P4_D4_N144_q6.4_v1',
                method=method, count=144, rate=6.4, layers=layers, suffix=2048,
                matched=True, hidden_source_slots=32, hidden_destination_slots=32,
                D_wait_for_source_macro=True, D_combine_ready_macros=False)
            if method == 'Token':
                cfg['token_suffix_by_request'] = matched_token_plan(layers, 144)
            configs.append(cfg)
    values = []
    try:
        for cfg in configs:
            value = job(cfg)
            if value is not None:
                values.append(value)
                write(ROOT / 'analysis/frontier_followup.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
