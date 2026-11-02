"""Compare properly warmed native baselines with identical durable evidence."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run,gate
from .primary_campaign import subset_parity
from .async_campaign import integrity


def make():
    revised=ROOT/'state/warm_epoch_plan_stats_retry_v3.json'
    if revised.exists():return json.loads(revised.read_text())
    destination=ROOT/'state/warm_epoch_plan.json'
    if destination.exists():return json.loads(destination.read_text())
    original=json.loads((ROOT/'state/async_epoch_plan.json').read_text());models=[]
    for item in original['models']:
        model=item['model']
        base=json.loads((ROOT/'configs'/(item['native_gate']+'.json')).read_text())
        base.update(measurement_epoch='WARM_GRAPH_DURABLE_V2',prime_all_graph_shapes=True,
            observe_graph_captures=True,verify_transfer=False,state_probe=False,timeout=900)
        record=dict(model=model,reference=item['reference'],byte_gate=None,output_gate=None,controls=[],pilots=[],DP_gates=[],DP_pilots=[])
        for count,heavy,key in [(2,True,'byte_gate'),(8,False,'output_gate')]:
            cfg=copy.deepcopy(base)
            cfg.update(name=f'warm_epoch_{model}_native_gate{count}',phase='correctness_warmed_native',
                state_probe=heavy,verify_transfer=heavy)
            record[key]=put(cfg,count)
        rate=2. if model=='ouro' else 1.
        for label,warmed,asynchronous_evidence in [('cold_sync',False,False),('cold_async',False,True),('warm_async',True,True)]:
            cfg=copy.deepcopy(base)
            cfg.update(name=f'graph_control_{model}_{label}_q{str(rate).replace(".","p")}_n24',
                prime_all_graph_shapes=warmed,async_evidence=asynchronous_evidence,phase='graph_warmup_control')
            record['controls'].append(put(cfg,24,rate))
        rates=[1.5,2.5,3.] if model=='ouro' else [.6,1.,1.2]
        for chunk in (base['length'],2048):
            for rate in rates:
                cfg=copy.deepcopy(base)
                cfg.update(name=f'warm_epoch_{model}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    chunk=chunk,phase='qps_pilot')
                record['pilots'].append(put(cfg,24,rate))
        for chunk in (512,2048):
            cfg=json.loads((ROOT/'configs'/f'async_aggregated_{model}_l{base["length"]}_c{chunk}_gate8.json').read_text())
            cfg.update(name=f'warm_aggregated_{model}_c{chunk}_gate8',measurement_epoch='WARM_GRAPH_DURABLE_V2',
                prime_all_graph_shapes=True,observe_graph_captures=True,phase='correctness_warmed_DP2',timeout=900)
            record['DP_gates'].append(put(cfg,8))
            for rate in rates:
                c=copy.deepcopy(cfg)
                c.update(name=f'warm_aggregated_{model}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',phase='aggregated_QPS_pilot')
                record['DP_pilots'].append(put(c,24,rate))
        models.append(record)
    result=dict(created=time.time(),models=models,
        scope='Both PD and Native Mixed DP2 keep the original full-prompt warm-up, then warm supported native graph row geometries with actual fixed-depth requests. Extra short prefixes are startup only. Measured prompts, arrivals, outputs and frozen SLOs do not change. No graph implementation, precision or model weight changes.')
    write(destination,result);return result


def main():
    ledger=[]
    for item in make()['models']:
        model=item['model']
        for key in ('byte_gate','output_gate'):
            name=item[key]
            write(ROOT/'state/warm_campaign.json',dict(stage='WARMED_CORRECTNESS',name=name,time=time.time()))
            okay=run(name) and subset_parity(item['reference'],name)
            if okay and json.loads((ROOT/'configs'/(name+'.json')).read_text()).get('async_evidence'):okay=integrity(name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                check=dict(work=evidence.full_work(name))
                if key=='byte_gate':check['transfer']=evidence.transfer(name)
                check['passed']=all(v['passed'] for v in check.values())
                write(ROOT/'raw'/name/'recovery_gate.json',check);okay=check['passed']
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/warm_results.json',ledger)
            if not okay:break
        if not okay:continue
        reference=None
        for name in item['controls']+item['pilots']:
            write(ROOT/'state/warm_campaign.json',dict(stage='WARMED_PD_MEASUREMENT',name=name,time=time.time()))
            okay=run(name) and subset_parity(item['reference'],name)
            if okay and json.loads((ROOT/'configs'/(name+'.json')).read_text()).get('async_evidence'):okay=integrity(name)
            if okay and reference:okay=subset_parity(reference,name)
            if reference is None:reference=name
            ledger.append(dict(name=name,passed=okay));write(ROOT/'state/warm_results.json',ledger)
            if not okay:break
        valid=[]
        for name in item['DP_gates']:
            okay=run(name) and subset_parity(item['output_gate'],name) and integrity(name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                check=dict(work=evidence.full_work(name))
                check['passed']=check['work']['passed']
                write(ROOT/'raw'/name/'recovery_gate.json',check);okay=check['passed']
            ledger.append(dict(name=name,passed=okay))
            if okay:valid.append(json.loads((ROOT/'configs'/(name+'.json')).read_text())['chunk'])
        for name in item['DP_pilots']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if cfg['chunk'] not in valid:continue
            write(ROOT/'state/warm_campaign.json',dict(stage='WARMED_AGGREGATED_MEASUREMENT',name=name,time=time.time()))
            okay=run(name) and subset_parity(item['output_gate'],name) and integrity(name)
            ledger.append(dict(name=name,passed=okay));write(ROOT/'state/warm_results.json',ledger)
            if not okay:break
    write(ROOT/'state/warm_campaign.json',dict(stage='WARM_EPOCH_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger))
    restore_if_idle()


if __name__=='__main__':main()
