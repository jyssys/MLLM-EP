"""Capture actual kernels from the final warmed configuration, not older controls."""
import copy
from .common import *
from .initial_campaign import run
from .cuda_only_profile import analyze
from .gates import one


def main():
    while True:
        path=ROOT/'state/confirmation.json'
        if (path.exists() and json.loads(path.read_text()).get('stage') in ('CONFIRMATION_PLAN_READY','REPLICATION','CONFIRMATION_COMPLETE_REVIEW_REQUIRED')
            and (ROOT/'state/confirmation_plan.json').exists()):break
        time.sleep(5)
    selection=json.loads((ROOT/'state/confirmation_plan.json').read_text())['selection']
    expanded=[]
    for item in selection:
        if not item.get('PD_pilot'):continue
        expanded.append(dict(item,variant='selected'))
        pressure=[]
        for path in (ROOT/'configs').glob('*.json'):
            c=json.loads(path.read_text())
            if not isinstance(c,dict) or c.get('model')!=item['model'] or c.get('phase')!='qps_pilot':continue
            if c.get('measurement_epoch')!='WARM_GRAPH_DURABLE_V2' or c.get('arm','').startswith('DP'):continue
            terminal=ROOT/'raw'/c['name']/'dispatcher_result.json'
            if not terminal.exists() or not json.loads(terminal.read_text()).get('complete'):continue
            g=one(c)
            pressure.append((c,g))
        if pressure:
            best=max(pressure,key=lambda v:((v[1].get('P_fraction_reference') or 0)>=.85 and (v[1].get('P_ready_lower_bound_half_growth') or 0)>0,
                min(v[1].get('P_fraction_reference') or 0,1.1),-(v[1].get('D_credit_refusal_sample_fraction') or 0)))
            if best[0]['name']!=item['PD_pilot']:expanded.append(dict(model=item['model'],PD_pilot=best[0]['name'],variant='pressure'))
    ledger=[]
    for item in expanded:
        clean=item.get('PD_pilot')
        if not clean:continue
        source=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
        for roles in (['prefill','decode'],['prefill']):
            cfg=copy.deepcopy(source)
            cfg.update(name=f'profile_warm_{item["model"]}_{item["variant"]}_{"both" if len(roles)==2 else "P_only"}_24',
                phase='bounded_stage_profiling',cuda_only_profile=True,capture_roles=roles,
                capture_role='prefill',P_only_profiler=False,nsys=True,matched_clean=clean,
                profile_finalization_timeout=600,timeout=900)
            cfg['native_transport_limits']=dict(cfg.get('native_transport_limits',{}),shutdown_timeout=600)
            write(ROOT/'configs'/(cfg['name']+'.json'),cfg)
            write(ROOT/'state/warm_profile.json',dict(stage='PROFILE_ACTIVE',name=cfg['name'],time=time.time()))
            okay=run(cfg['name'])
            record=dict(name=cfg['name'],complete=okay)
            if okay:
                record['roles']={role:analyze(cfg['name'],clean,role) for role in roles}
                record['same_start_gates']=one(cfg)
            ledger.append(record);write(ROOT/'state/warm_profile_results.json',ledger)
            if okay and record['roles']['prefill'].get('eligible_for_matched_clean'):break
    write(ROOT/'state/warm_profile.json',dict(stage='WARM_PROFILE_COMPLETE_REVIEW_REQUIRED',results=ledger,time=time.time()))
    restore_if_idle()


if __name__=='__main__':main()
