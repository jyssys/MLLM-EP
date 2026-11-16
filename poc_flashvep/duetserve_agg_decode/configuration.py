"""Keep QPS, token budget, sequence limit and graph mode independently controlled."""
import random

from .common import PRIOR, ROOT, save
import json


def offsets(count, rate, seed):
    rng = random.Random(seed)
    values = [0.0]
    for _ in range(count - 1):
        values.append(values[-1] + rng.expovariate(rate))
    return values


def standard_capture_sizes(slots):
    """Match this vLLM version's throughput-mode default, without sparse padding."""
    maximum = min(2 * slots, 512)
    return [n for n in (1, 2, 4) if n <= maximum] + list(range(8, min(maximum + 1, 256), 8)) + list(range(256, maximum + 1, 16))


def base(name, budget=8192, slots=16, rate=4.0, count=64, cap=200,
         seed=2026111601, arm='DP2', **changes):
    template = PRIOR / 'configs/boundary_C_FULL_B512_S16_ROUND_ROBIN_O16_q5.2_s2026111459_W600_v1.json'
    config = json.loads(template.read_text())
    for key in ('fixed_measurement', 'prompt_reuse_scope', 'selection_SHA256'):
        config.pop(key, None)
    config.update(name=name, arm=arm, evaluation_arm='Colocated' if arm == 'DP2' else 'Native',
        count=count, cap=cap, dataset='discovery.json', request_caps=[cap] * count,
        arrival_offsets=offsets(count, rate, seed), arrival_rate=rate, arrival_seed=seed,
        token_budget=budget, chunk=budget, D_slots=slots,
        ar_cudagraph_mode='FULL', ar_compile_mode=0, ar_async_scheduling=True,
        gpu_phase_mask_norm=True, direct_phase_norm=False, opaque_phase_norm=False,
        phase_stable_norm=True, colocated_cohorts=False, placement='ROUND_ROBIN',
        warmup_batch_sizes=[1, 2, 4], warmup_output_cap=16, warmup_count=2,
        phase='DUETSERVE_FIXED_CONFIGURATION_CHARACTERIZATION',
        physical_activity=False, execution_events=False, boundary_events=False,
        full_prompt_state_witness=False, hidden_verify_hash=False,
        verify_layer_execution=False, clean_timing=True, correctness_only=False,
        timeout=1200, wall_timeout=1800, module='native', policy=None,
        cost_model=None, canonical_new_KV_block_order=False,
        coordinator_GIL_switch_interval_s=0.001,
        scope='New measurement. Frozen scheduler settings across QPS. No sustainable claim from a finite cohort.')
    if arm != 'DP2':
        config.update(token_budget=32000, chunk=8000, atomic_prompt=True, D_token_budget=128000,
            serialize_D_imports=True, D_admission_batch=16, D_cohort_collect_ms=2,
            cohort_barrier=True, canonical_new_KV_block_order=True,
            reuse_prefill_token=True, async_decode=True, ar_async_scheduling=False,
            ar_cudagraph_mode='FULL_DECODE_ONLY', gpu_phase_mask_norm=False,
            direct_phase_norm=True, nixl_manual_coalesce=True, nixl_descriptor_sort=False)
    config.update(changes)
    pool = json.loads((ROOT / 'data' / config['dataset']).read_text())
    if count > len(pool):
        expanded = [pool[i % len(pool)] for i in range(count)]
        filename = 'declared_cycle_' + str(count) + '_' + config['dataset']
        path = ROOT / 'data' / filename
        if path.exists():
            assert json.loads(path.read_text()) == expanded
        else:
            from .common import write
            write(path, expanded)
        config.update(dataset=filename, prompt_reuse_scope='Declared real-text pool cycle. No prefix caching. Cycles are not independent prompts.')
    return config


if __name__ == '__main__':
    from .common import initialize
    initialize()
    for budget in (8192, 512):
        name = f'qual_DP2_B{budget}_S16_FULL_N32_O200_q4_s2026111601_v1'
        save(base(name, budget=budget, count=32, rate=4.0, correctness_only=True))
        print(name)
