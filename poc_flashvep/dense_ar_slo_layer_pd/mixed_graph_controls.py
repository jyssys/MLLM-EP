"""Give controlled burst and mixed-output studies a qualified strong graph arm."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def main():
    gate=ROOT/'analysis/compiled_opaque_norm_gate.json'
    assert json.loads(gate.read_text())['passed']
    sources=['mixed_discovery_BURSTY_O16_NATIVE_L0_F0.0_q2.0_N256_v1',
        'mixed_discovery_MIXED_OUTPUT_NATIVE_L0_F1.0_q1.0_N128_v1']
    try:
        for source in sources:
            path=ROOT/'configs'/(source+'.json');limit=time.time()+14400
            while not path.exists():
                assert time.time()<limit;time.sleep(10)
            reference=json.loads(path.read_text());name='strong_graph_'+source
            cfg=base(name,'Colocated',count=reference['count'],cap=reference['cap'],
                rate=reference['arrival_rate'],dataset=reference['dataset'],seed=reference['arrival_seed'],
                arrival_offsets=reference['arrival_offsets'],request_caps=reference['request_caps'],
                chunk=512,token_budget=512,D_slots=16,verified_ByteLevel_priming=True,
                ar_async_scheduling=True,gpu_phase_mask_norm=True,opaque_unfused_norm=True,
                ar_compile_mode=3,ar_cudagraph_mode='FULL',startup_timeout=900,wall_timeout=2400,
                ar_execution_family='COMPILE3_OPAQUE_ORIGINAL_UNFUSED_NORM',
                explicit_cudagraph_capture_sizes=[1,2,4,8,16,256,512],
                phase='QUALIFIED_STRONG_GRAPH_MIXED_BURST_DISCOVERY',
                paired_source_configuration=source,
                actual_controlled_burst_seed=2026111461 if 'BURSTY' in source else None,
                controlled_output_distribution_seed=2026111461 if 'MIXED_OUTPUT' in source else None)
            for key in ('arrival_distribution','output_distribution'):
                if key in reference:cfg[key]=reference[key]
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('MIXED_STRONG_GRAPH',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
