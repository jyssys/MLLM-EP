"""Keep the final Static comparison strong without an exhaustive grid."""
import fcntl
from .common import *
from .configuration import base
from .expired_deadline_recovery import options
from .run import execute
from .qualification import against_previous
from .analyze import metrics
from .sustainable import duration_config,assess


def settings(fraction):
    value=options(False);margin=json.loads((ROOT/'configs/canonical_completion_residual_v1.json').read_text())['total_margin_ms']
    value.update(canonical_new_KV_block_order=True,intern_immutable_prompt_pool=True,
        cost_model='canonical_cost_model_v5.json',D_admission_batch=4,static_layers=12,
        static_request_block_size=8,migration_fraction=fraction,safety_margin_ms=margin,
        safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)},
        phase='FINAL_STRONG_STATIC_FRACTION_DISCOVERY_NOT_HELD_OUT')
    return value


def main():
    try:
        gate=ROOT/'analysis/canonical_capacity_Static12_D4_O16_q4.8_s2026111462_W600_v1_sustainable_v2.json'
        while not gate.exists():time.sleep(10)
        values=[]
        for tag,fraction in (('all',1.),('f09375',.9375),('f0875',.875)):
            name=f'canonical_STATIC_fraction_{tag}_N128_O16_q4.8_v1'
            cfg=base(name,'STATIC',count=128,cap=16,rate=4.8,**settings(fraction))
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            values.append(dict(name=name,fraction=fraction,metrics=value))
            print('STATIC_FRACTION_SHORT',tag,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
        baseline=values[0]['metrics']
        eligible=[v for v in values[1:] if v['metrics']['TTFT_p95_ms']<=.95*baseline['TTFT_p95_ms']
            and v['metrics']['ITL_p99_ms']<=1.05*baseline['ITL_p99_ms']
            and v['metrics']['SLO_regimes']['Decode_priority']['Joint_attainment']>=.95]
        selected=None
        if eligible:
            selected=min(eligible,key=lambda v:v['metrics']['TTFT_p95_ms'])
            name=f"canonical_STATIC_fraction_long_f{str(selected['fraction']).replace('.','p')}_O16_q4.8_s2026111462_W600_v1"
            cfg=duration_config(name,'STATIC',4.8,seed=2026111462,warm=60.,measure=600.,follow=90.,
                **settings(selected['fraction']))
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            long=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
            selected=dict(name=name,fraction=selected['fraction'],qualified=long['SLO_regimes']['Decode_priority']['qualified'])
        write(ROOT/'analysis/final_Static_fraction_discovery.json',dict(time=time.time(),short_runs=values,
            long_selected=selected,evidence='MEASURED',
            scope='Discovery only. Test a fraction in a long window only if its short TTFT improves at least5%, its Decode tail stays within5% of the shared Static reference, and joint attainment reaches95%. No hidden confirmation data selects the fraction. Exact work, output and lifetime remain required.'))
        write(ROOT/'state/final_Static_fraction_discovery_complete.json',dict(time=time.time(),selected=selected))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
