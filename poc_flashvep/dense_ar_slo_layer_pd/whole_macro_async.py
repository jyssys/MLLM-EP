"""Test qualified larger cohorts after the async Decode intervention."""
import fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .stages import state_gate
from .residual_guard_study import calibrate
from .sustainable import duration_config,assess
from .analyze import metrics


def main():
    options=method_settings(wait=True)
    options.update(async_decode=True,cost_model='coalesced_cost_model_v2.json',
        separate_nominal_objective=True,phase='WHOLE_MACRO_ASYNC_GEOMETRY_DISCOVERY')
    try:
        name='qual_whole_macro_async_G8_D8_N32_full_state_v1'
        cfg=base(name,'FORCED_QUALIFICATION',count=32,cap=16,request_caps=[2]*32,
            arrival_offsets=[0.]*32,arrival_rate=None,forced_layers=[2,4,8,12],
            verify_layer_execution=True,hidden_verify_hash=True,
            full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill','decode'],
            correctness_only=True,clean_timing=False,negotiation_timeout_s=120,**options)
        assert execute(save(cfg),restore=False);output=against_previous(name);assert output['passed']
        work=state_gate(name,32);assert work['passed'];expected=independent()
        actual=json.loads((ROOT/'raw'/name/'full_prompt_state_decode.json').read_text())['requests']
        checks={rid:dict(KV_equal=value['KV']==expected[rid]['KV'],
            Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
            for rid,value in actual.items()}
        passed=len(checks)==32 and all(all(v.values()) for v in checks.values())
        write(ROOT/'analysis/whole_macro_async_full_state_gate.json',dict(name=name,passed=passed,
            output=output,work=work,checks=checks,evidence='CORRECTNESS_ONLY'))
        assert passed
        # Native0 already has the separate async imported-KV qualification.
        for arm in ('STATIC','DIRECT'):
            changes=dict(options,static_layers=12,migration_fraction=1.)
            if arm=='DIRECT':
                margin=calibrate()['total_margin_ms']
                changes.update(pipeline_objective_excludes_existing_wait=True,
                    queue_benefit_external_only=True,safety_margin_ms=margin,
                    safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)})
            name=f'whole_macro_async_{arm}_L12_G8_D8_N128_O16_q5.0_v1'
            cfg=base(name,arm,count=128,cap=16,rate=5.,**changes)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('WHOLE_MACRO_PILOT',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
            name=f'whole_macro_async_{arm}_L12_G8_D8_O16_q5.0_s2026111459_W600_v1'
            cfg=duration_config(name,arm,5.,seed=2026111459,warm=60.,measure=600.,follow=90.,**changes)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
            print('WHOLE_MACRO_LONG',name,value['raw_completed_output_tokens_s'],
                {k:v['qualified'] for k,v in value['SLO_regimes'].items()},flush=True)
        write(ROOT/'state/whole_macro_async_discovery_complete.json',dict(time=time.time()))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
