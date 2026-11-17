"""Do not compare layer offload only against a serialized weak token path."""
import fcntl
import json
from .common import ROOT, SLO_PRIOR, QUIET_LOCK, resources, save, write
from .run import execute
from .analysis import metrics


def configuration(name, method='Token', count=64, cap=16, budget=512,
                  suffix=2048, continuous=True, correctness=False, **changes):
    from ..duetserve_agg_decode import configuration as inherited
    inherited.ROOT, inherited.PRIOR = ROOT, SLO_PRIOR
    cfg = inherited.base(name, arm='PD_G1', rate=4.8, count=count, cap=cap, seed=2026111710,
        entry_module='poc_flashvep.dense_ar_layer_token_pd.serving',
        evaluation_arm='StandardToken' if method == 'Token' else 'StandardNative',
        D_token_budget=budget, native_continuous_D_ingress=continuous,
        cohort_barrier=False if continuous else True,
        D_admission_batch=16, D_cohort_collect_ms=0 if continuous else 2,
        ar_cudagraph_mode='FULL' if continuous else 'FULL_DECODE_ONLY',
        gpu_phase_mask_norm=continuous, direct_phase_norm=not continuous,
        explicit_cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if continuous else [1, 2, 4, 8, 16],
        warmup_batch_sizes=[1, 2, 3, 4], warmup_count=2,
        correctness_only=correctness, clean_timing=not correctness,
        runtime_observation=correctness, verify_layer_execution=correctness)
    if method == 'Token':
        cfg.update(token_axis=True, token_suffix=suffix, reuse_prefill_token=False,
            D_prefill_chunk=budget if continuous else 8000,
            axis_state_witness=correctness, axis_post_runner_witness=correctness)
    cfg.update(changes)
    return cfg


def main():
    try:
        from .qualification import gate
        name = 'standard_Token_continuous_B512_state_N8_O64_v1'
        cfg = configuration(name, count=8, cap=64, correctness=True,
                            warmup_count=0, warmup_batch_sizes=[])
        assert execute(save(cfg))
        assert gate(name, 'Native_state_N8_O16_v1')
        name = 'standard_Token_continuous_B512_state_N32_O64_v1'
        cfg = configuration(name, count=32, cap=64, correctness=True,
                            warmup_count=0, warmup_batch_sizes=[])
        assert execute(save(cfg))
        assert gate(name, 'Native_state_N32_O16_v1')
        jobs = [configuration(f'frontier_standard_Token_B{b}_S2048_N64_O16_q4.8_v1', budget=b)
                for b in (256, 512, 1024, 8192)]
        jobs.extend(configuration(f'frontier_standard_Token_serial_S{s}_N64_O16_q4.8_v1',
                                  suffix=s, budget=128000, continuous=False) for s in (2048, 4096))
        jobs.extend(configuration(f'frontier_standard_Native_{mode}_N64_O16_q4.8_v1', method='Native',
                                  budget=128000, continuous=mode == 'continuous')
                    for mode in ('serial', 'continuous'))
        values = []
        for cfg in jobs:
            if not execute(save(cfg)):
                continue
            value = metrics(cfg['name'])
            assert value['output_gate']['passed'], value['output_gate']
            values.append(value)
            write(ROOT / 'analysis/standard_token_native_frontier_discovery.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
