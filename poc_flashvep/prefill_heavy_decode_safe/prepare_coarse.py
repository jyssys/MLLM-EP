"""Choose overloaded discovery points from measured finite P references."""
import copy,json,random
from .common import *
from .capacity_analysis import analyze
from .gates import model_contract


def native_configuration(model,length,cap,count,rate,seed,dataset,slots=1):
    anchor=f'anchor_{model}_L{4096 if model=="ouro" else 2048}_O{32 if model=="ouro" else 16}_clean_cupti26_N64_a2026111210'
    cfg=json.loads((ROOT/'configs'/(anchor+'.json')).read_text());pool=json.loads((ROOT/'data'/dataset).read_text())
    assert len(pool)>=count and len({r['prompt_id'] for r in pool[:count]})==count
    rng=random.Random(seed);arrivals=[0.]
    for _ in range(count-1):arrivals.append(arrivals[-1]+rng.expovariate(rate))
    cfg.update(length=length,cap=cap,count=count,state_capacity_tokens=length,chunk=length,token_budget=slots*length,
        P_slots=slots,P_prefill_graph_rows=slots*length,D_prefill_graph_rows=length,D_token_budget=length,
        D_tile_tokens=length,dataset=dataset,prompt_ids=[r['prompt_id'] for r in pool[:count]],request_caps=[cap]*count,
        arrival_rate=rate,arrival_seed=seed,arrival_offsets=arrivals,workload=f'L{length}_O{cap}',
        discovery_activity=False,discovery_event_observer=False,state_probe=False,verify_transfer=False,
        async_prompt_state_probe=False,async_verify_handoff=False,observe_service_timing=False,
        observe_device_execution=False,required_qualification=f'native_{model}_{"early_profile" if model=="ouro" else "clean_recovery"}_state32',
        phase='MEASURED_CAPACITY_GUIDED_NATIVE_DISCOVERY',timeout=900,startup_timeout=300,
        data_scope=dict(distinct_prompt_ids=count,pool_records=len(pool),prefix_caching=False,cyclic_reuse=False,
            scope='Controlled real-document prefixes. New Poisson arrival seed. Discovery, not held-out confirmation.'))
    assert model_contract(model,length,cap)
    anchor_slo=json.loads((ROOT/'configs/frozen_SLO.json').read_text())[f'{model}_L{4096 if model=="ouro" else 2048}']
    cfg['campaign_frozen_SLO']={k:anchor_slo[k] for k in ('TTFT_ms','TPOT_ms','ITL_ms')}
    cfg['SLO_scope']='The original anchor thresholds remain unchanged at every new length. No length-specific relaxation.'
    return cfg


def main():
    initialize();jobs=[];references=[]
    candidates=[('ouro',16384,'natural_discovery',True),('huginn',3900,'natural_discovery',True),
        ('ouro',8192,'confirmation',True),('huginn',3500,'confirmation',False),('huginn',3072,'confirmation',False)]
    for model,length,split,long in candidates:
        reference=f'capacity_{model}_L{length}_{"O16_" if long else ""}B124_start1'
        value=analyze(reference);best=value['strongest_finite_P_reference'];assert best is not None,reference
        rate=1.15*best['P_requests_per_s']['p50'];cfg=native_configuration(model,length,16,64,rate,2026111230,
            f'{split}_{model}_L{length}.json',slots=1)
        cfg.update(name=f'coarse_{model}_L{length}_O16_P1_mu115_N64_a2026111230',
            capacity_reference=reference,capacity_fraction_offered=1.15,
            selection_reason='Start above the strongest measured finite P service rate. P1 is not an artificial slowdown when measured B1 is the strongest cohort. Online queue, KV, kernel, and SLO gates still decide.')
        save(cfg);jobs.append(cfg['name']);references.append(value)
    write(ROOT/'analysis/coarse_capacity_references.json',references)
    write(ROOT/'state/coarse_discovery_plan.json',dict(jobs=jobs,
        scope='The first candidate per length is capacity-guided, not a fixed QPS claim. All primary outputs remain at least sixteen. Later decisions depend on measured bottlenecks.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
