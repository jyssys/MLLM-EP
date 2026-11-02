"""Remove foreground fsync interference without removing any evidence record."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run,gate
from .primary_campaign import subset_parity
from .search_plan import make as primary_plan


def integrity(name):
    folder=ROOT/'raw'/name;proof=[]
    for path in folder.glob('async_evidence_summary_*.json'):
        value=json.loads(path.read_text())
        assert not value['error'] and value['enqueued_records']==value['background_written_records']
        proof.append(value)
    result=dict(name=name,writer_processes=len(proof),passed=bool(proof),
        enqueued=sum(v['enqueued_records'] for v in proof),durably_written=sum(v['background_written_records'] for v in proof),
        foreground_enqueue_seconds=sum(v['foreground_enqueue_seconds'] for v in proof),
        max_foreground_enqueue_seconds=max((v['foreground_max_enqueue_seconds'] for v in proof),default=None),
        scope='Process lifetime includes warm-up. Every original append record is retained and flushed before terminal exit. Background I/O durations are not additive E2E components.')
    write(folder/'async_evidence_integrity.json',result)
    return result['passed']


def make():
    target=ROOT/'state/async_epoch_plan.json'
    if target.exists():return json.loads(target.read_text())
    models=[]
    for record in primary_plan()['primaries']:
        model=record['model'];length=record['length'];cap=record['cap']
        base=json.loads((ROOT/'configs'/(record['gate'][0]+'.json')).read_text())
        base.update(timeout=900,pd_shutdown_timeout=300,exact_descriptor_packing=True,
            native_transport_limits=dict(base.get('native_transport_limits',{}),max_inflight_bytes=8*1024**3),
            execution_control_family='ADMISSION_RECOVERY_DURABLE_EVIDENCE',
            measurement_epoch='ASYNC_DURABLE_V1',async_evidence=True,workload=f'L{length}_O{cap}',
            observe_descriptor_cost=False,record_descriptor_replay=False)
        for key in ('capacity_reference','reference_capacity','rate_factor','rate_scope'):base.pop(key,None)
        cfg=copy.deepcopy(base)
        cfg.update(name=f'async_epoch_{model}_l{length}_native_gate8',phase='correctness_async_evidence')
        native_gate=put(cfg,8)
        item=dict(model=model,reference=record['gate'][0],native_gate=native_gate,controls=[],pilots=[],DP_gates=[],DP_pilots=[])
        rate=2. if model=='ouro' else 1.
        for enabled in (False,True):
            cfg=copy.deepcopy(base)
            label='async' if enabled else 'sync'
            cfg.update(name=f'logging_control_{model}_{label}_q{str(rate).replace(".","p")}_n24',
                phase='logging_transport_control',async_evidence=enabled,state_probe=False,verify_transfer=False)
            item['controls'].append(put(cfg,24,rate))
        rates=[1.5,2.,2.5,3.] if model=='ouro' else [.6,.8,1.,1.2]
        for chunk in (length,2048):
            for rate in rates:
                cfg=copy.deepcopy(base)
                cfg.update(name=f'async_epoch_{model}_l{length}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    chunk=chunk,phase='qps_pilot',verify_transfer=False,state_probe=False)
                item['pilots'].append(put(cfg,24,rate))
        for chunk in (512,2048):
            cfg=json.loads((ROOT/'configs'/f'aggregated_{model}_l{length}_c{chunk}_gate8.json').read_text())
            cfg.update(name=f'async_aggregated_{model}_l{length}_c{chunk}_gate8',async_evidence=True,
                measurement_epoch='ASYNC_DURABLE_V1',phase='correctness_async_DP2',timeout=600)
            item['DP_gates'].append(put(cfg,8))
            for rate in rates:
                c=copy.deepcopy(cfg)
                c.update(name=f'async_aggregated_{model}_l{length}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    phase='aggregated_QPS_pilot')
                item['DP_pilots'].append(put(c,24,rate))
        models.append(item)
    result=dict(created=time.time(),models=models,
        evidence='PROSPECTIVE_CONTROL_AFTER_MEASURED_FOREGROUND_FSYNC_STALLS',
        scope='The paired logging controls change only evidence transport. The new serving epoch applies the same durable writer to PD and true Native Mixed DP2. Supported 8-GiB credit and exact CPU descriptor packing are generic baseline tuning. No speedup from these changes is loop-specific. Frozen SLOs do not change.')
    write(target,result)
    return result


def main():
    ledger=[]
    for item in make()['models']:
        model=item['model']
        while not (ROOT/'configs/frozen_SLO.json').exists() or model+'_L'+str(6144 if model=='ouro' else 2560) not in json.loads((ROOT/'configs/frozen_SLO.json').read_text()):time.sleep(5)
        write(ROOT/'state/async_campaign.json',dict(stage='ASYNC_NATIVE_CORRECTNESS',model=model,time=time.time()))
        okay=run(item['native_gate']) and gate(item['native_gate'],item['reference']) and integrity(item['native_gate'])
        ledger.append(dict(name=item['native_gate'],passed=okay))
        if not okay:
            write(ROOT/'state'/f'async_blocked_{model}.json',dict(time=time.time(),gate=item['native_gate']))
            continue
        reference=None
        for name in item['controls']+item['pilots']:
            write(ROOT/'state/async_campaign.json',dict(stage='ASYNC_PD_MEASUREMENT',name=name,time=time.time()))
            okay=run(name) and subset_parity(item['reference'],name)
            if okay and reference:okay=subset_parity(reference,name)
            if reference is None:reference=name
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if okay and cfg.get('async_evidence'):okay=integrity(name)
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/async_results.json',ledger)
            if not okay:break
        valid=[]
        for name in item['DP_gates']:
            okay=run(name) and subset_parity(item['native_gate'],name) and integrity(name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                okay=evidence.full_work(name)['passed']
            ledger.append(dict(name=name,passed=okay))
            if okay:valid.append(json.loads((ROOT/'configs'/(name+'.json')).read_text())['chunk'])
        for name in item['DP_pilots']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if cfg['chunk'] not in valid:continue
            write(ROOT/'state/async_campaign.json',dict(stage='ASYNC_AGGREGATED_MEASUREMENT',name=name,time=time.time()))
            okay=run(name) and subset_parity(item['native_gate'],name) and integrity(name)
            ledger.append(dict(name=name,passed=okay));write(ROOT/'state/async_results.json',ledger)
            if not okay:break
    write(ROOT/'state/async_campaign.json',dict(stage='ASYNC_EPOCH_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger))
    restore_if_idle()


if __name__=='__main__':main()
