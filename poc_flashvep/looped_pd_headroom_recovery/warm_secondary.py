"""Test longer validated prompts after generic tail controls, before replication."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run
from .primary_campaign import subset_parity
from .async_campaign import integrity
from .gates import one


def main():
    while True:
        warm=ROOT/'state/warm_campaign.json';ladder=ROOT/'state/adaptive_ladder.json'
        if (warm.exists() and json.loads(warm.read_text()).get('stage')=='WARM_EPOCH_COMPLETE_REVIEW_REQUIRED'
            and ladder.exists() and json.loads(ladder.read_text()).get('stage')=='ADAPTIVE_LADDER_COMPLETE_REVIEW_REQUIRED'):break
        time.sleep(5)
    from .warm_campaign import make
    primary=make();ledger=[];plans=[]
    for model,length,cap,rates in [('ouro',8192,32,[2.,2.5,3.]),('huginn',3072,32,[.8,1.,1.2])]:
        item=next(r for r in primary['models'] if r['model']==model)
        ref=f'secondary_{model}_l{length}_o{cap}_c{length}_gate8'
        accepted=ROOT/'raw'/ref/'recovery_gate.json'
        if not accepted.exists() or not json.loads(accepted.read_text()).get('passed'):
            ledger.append(dict(model=model,status='NOT_RUN_NO_VALID_LONGER_PROMPT_GATE'));continue
        base=json.loads((ROOT/'configs'/(ref+'.json')).read_text())
        base.update(async_evidence=True,exact_descriptor_packing=True,prime_all_graph_shapes=True,
            observe_graph_captures=True,measurement_epoch='WARM_GRAPH_DURABLE_V2',state_probe=False,
            verify_transfer=False,timeout=900)
        valid=True;last=None
        for count,heavy in [(2,True),(8,False)]:
            c=copy.deepcopy(base)
            c.update(name=f'warm_secondary_{model}_l{length}_byte_gate2' if heavy else f'warm_secondary_{model}_l{length}_output_gate8',
                phase='correctness_warmed_secondary',state_probe=heavy,verify_transfer=heavy)
            name=put(c,count);okay=run(name) and subset_parity(ref,name) and integrity(name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                checks=dict(work=evidence.full_work(name))
                if heavy:checks['transfer']=evidence.transfer(name)
                checks['passed']=all(v['passed'] for v in checks.values())
                write(ROOT/'raw'/name/'recovery_gate.json',checks);okay=checks['passed']
            ledger.append(dict(name=name,passed=okay));write(ROOT/'state/warm_secondary_results.json',ledger)
            if not okay:valid=False;break
            last=name
        if not valid:continue
        pilots=[]
        for chunk in (length,2048):
            for rate in rates:
                c=copy.deepcopy(base)
                c.update(name=f'warm_secondary_{model}_l{length}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',chunk=chunk,phase='qps_pilot')
                name=put(c,24,rate)
                write(ROOT/'state/warm_secondary.json',dict(stage='LONGER_PROMPT_PILOT',name=name,time=time.time()))
                okay=run(name) and subset_parity(last,name) and integrity(name)
                ledger.append(dict(name=name,passed=okay));write(ROOT/'state/warm_secondary_results.json',ledger)
                if okay:pilots.append(c)
                else:break
        if not pilots:continue
        def score(c):
            g=one(c)
            return (bool(g['D_SLO_gate'] and g['D_credit_gate']),
                g['D_credit_refusal_sample_fraction'] is not None and g['D_credit_refusal_sample_fraction']<.1,
                (g['P_ready_lower_bound_half_growth'] or 0)>0,min(g.get('P_fraction_reference') or 0,1.1),
                -(g.get('D_token_ITL_p99_whole_run_ms') or 1e9))
        selected=max(pilots,key=score);rate=selected['arrival_rate'];dp=[]
        for chunk in (512,2048):
            c=json.loads((ROOT/'configs'/(item['DP_gates'][0]+'.json')).read_text())
            c.update(name=f'warm_secondary_aggregated_{model}_l{length}_c{chunk}_gate8',
                length=length,cap=cap,chunk=chunk,state_capacity_tokens=length,
                phase='correctness_warmed_secondary_DP2')
            if model=='huginn':
                # Four legal 3072/32 reservations need 24,832 blocks.
                # The old 24,576-block pool would cap this four-slot replica
                # at three requests despite unused physical HBM.
                c.update(kv_blocks=25600,KV_tuning_scope='Supported pool increase for four long requests plus native watermark. No overcommit. Requalify both chunk budgets before timing.')
            name=put(c,8)
            okay=run(name) and subset_parity(last,name) and integrity(name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                check=dict(work=evidence.full_work(name))
                check['passed']=check['work']['passed']
                write(ROOT/'raw'/name/'recovery_gate.json',check);okay=check['passed']
            ledger.append(dict(name=name,passed=okay))
            if not okay:continue
            c=copy.deepcopy(c)
            c.update(name=f'warm_secondary_aggregated_{model}_l{length}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',phase='aggregated_QPS_pilot')
            name=put(c,24,rate)
            okay=run(name) and subset_parity(last,name) and integrity(name)
            ledger.append(dict(name=name,passed=okay));dp.append(name)
            write(ROOT/'state/warm_secondary_results.json',ledger)
        plans.append(dict(model=model,length=length,cap=cap,PD_selected=selected['name'],DP_tuned=dp,
            scope='Two legal PD chunk budgets receive the same three rates. Both DP2 chunk budgets are tested at the selected matching rate. This is exploratory selection, not a strict positive result.'))
    write(ROOT/'state/warm_secondary.json',dict(stage='WARM_SECONDARY_COMPLETE_REVIEW_REQUIRED',results=ledger,plans=plans,time=time.time()))
    restore_if_idle()


if __name__=='__main__':main()
