"""Compare updated methods on identical controlled heterogeneous traces."""
import fcntl,random
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .objective_rate_discovery import graph_options
from .residual_guard_study import calibrate
from .workloads import modulated_arrivals
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def main():
    assert json.loads((ROOT/'analysis/whole_macro_async_full_state_gate.json').read_text())['passed']
    margin=calibrate()['total_margin_ms']
    try:
        for kind in ('BURSTY','MIXED_OUTPUT'):
            mixed=kind=='MIXED_OUTPUT';cap=200 if mixed else 16;rate=1. if mixed else 2.;count=128 if mixed else 256
            rng=random.Random(2026111461)
            caps=[rng.choices([16,64,200],[.6,.3,.1])[0] for _ in range(count)] if mixed else [16]*count
            arrivals=None if mixed else modulated_arrivals(rate,10*count/rate,2026111461)[:count]
            for label,arm,layers in (('Native','Native',0),('Static8','STATIC',8),
                    ('Static12','STATIC',12),('Dynamic','DIRECT',0),('Colocated','Colocated',0)):
                if arm=='Colocated':options=graph_options(512)
                elif arm=='Native':
                    options=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,async_decode=True,
                        nixl_manual_coalesce=True,nixl_descriptor_sort=False)
                else:
                    options=method_settings(wait=True)
                    options.update(async_decode=True,cost_model='coalesced_cost_model_v2.json',
                        separate_nominal_objective=True,static_layers=layers,migration_fraction=1.)
                    if arm=='DIRECT':
                        options.update(pipeline_objective_excludes_existing_wait=True,queue_benefit_external_only=True,
                            safety_margin_ms=margin,safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)},
                            expired_ttft_policy='QUEUE_RECOVERY')
                options.update(request_caps=caps,phase='UPDATED_ASYNC_HETEROGENEOUS_BURST_DISCOVERY',
                    actual_controlled_burst_seed=2026111461 if not mixed else None,
                    controlled_output_distribution_seed=2026111461 if mixed else None)
                if mixed:
                    options['output_distribution']=dict(caps=[16,64,200],probabilities=[.6,.3,.1],
                        scope='Controlled output caps, not Natural EOS.')
                else:
                    options.update(arrival_offsets=arrivals,arrival_distribution=dict(kind='PIECEWISE_POISSON',
                        period_s=40.,high_s=8.,high_multiplier=3.,low_multiplier=.5))
                name=f'async_updated_{kind}_{label}_N{count}_q{rate}_v1'
                if arm=='DIRECT':name=name.removesuffix('_v1')+'_recovery_v2'
                # Mixed outputs keep the original Poisson arrival trace. The
                # burst trace records its actual generator seed explicitly.
                cfg=base(name,arm,count=count,cap=cap,rate=rate,
                    seed=2026111421 if mixed else 2026111461,**options)
                assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
                value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
                print('ASYNC_MIXED',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
        write(ROOT/'state/async_mixed_discovery_complete.json',dict(time=time.time(),
            scope='One discovery start per controlled trace. This is not independent confirmation, public-trace generalization, or sustainable capacity.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
