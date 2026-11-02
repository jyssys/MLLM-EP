"""Check the central-window diagnosis on 128 requests, not only short pilots."""
import copy
from .common import *
from .initial_campaign import run
from .cuda_only_profile import analyze
from .gates import one


def main():
    while True:
        p=ROOT/'state/confirmation.json'
        if p.exists() and json.loads(p.read_text()).get('stage')=='CONFIRMATION_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    plan=json.loads((ROOT/'state/confirmation_plan.json').read_text());ledger=[]
    for model in ('ouro','huginn'):
        eligible_for_more=False
        rows=sorted([r for r in plan['runs'] if r['model']==model],key=lambda r:r['start'])
        for item in rows:
            if item['start']>1 and not eligible_for_more:
                ledger.append(dict(model=model,start=item['start'],status='NOT_RUN_NO_JOINT_FIRST_START_GATE'))
                continue
            clean=item['PD'];terminal=ROOT/'raw'/clean/'dispatcher_result.json'
            accounting=ROOT/'raw'/clean/'confirmation_work.json'
            if not terminal.exists() or not json.loads(terminal.read_text()).get('complete'):
                ledger.append(dict(model=model,start=item['start'],status='NOT_RUN_NO_COMPLETE_CLEAN_REFERENCE'));break
            if not accounting.exists() or not json.loads(accounting.read_text()).get('passed'):
                ledger.append(dict(model=model,start=item['start'],status='NOT_RUN_NO_VALID_FIXED_DEPTH_REFERENCE'));break
            source=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
            for roles in (['prefill','decode'],['prefill']):
                cfg=copy.deepcopy(source)
                cfg.update(name=f'long_profile_{model}_PD_start{item["start"]}_{"both" if len(roles)==2 else "P_only"}_n128',
                    phase='bounded_stage_profiling',cuda_only_profile=True,capture_roles=roles,capture_role='prefill',
                    P_only_profiler=False,nsys=True,matched_clean=clean,profile_finalization_timeout=1200,timeout=1800)
                cfg['native_transport_limits']=dict(cfg.get('native_transport_limits',{}),shutdown_timeout=1200)
                write(ROOT/'configs'/(cfg['name']+'.json'),cfg)
                write(ROOT/'state/long_window_profile.json',dict(stage='LONG_WINDOW_PROFILE',name=cfg['name'],time=time.time()))
                okay=run(cfg['name']);record=dict(name=cfg['name'],model=model,start=item['start'],complete=okay)
                if okay:
                    record['profiles']={role:analyze(cfg['name'],clean,role) for role in roles}
                    record['same_start_gates']=one(cfg)
                    g=record['same_start_gates']
                    eligible_for_more=bool(g.get('P_compute_gate') and g.get('D_SLO_gate') and g.get('D_credit_gate'))
                ledger.append(record);write(ROOT/'state/long_window_profile_results.json',ledger)
                if okay and record['profiles']['prefill'].get('eligible_for_matched_clean'):break
    write(ROOT/'state/long_window_profile.json',dict(stage='LONG_WINDOW_PROFILE_COMPLETE_REVIEW_REQUIRED',results=ledger,time=time.time(),
        scope='A short cohort can bunch Prefill completions outside its central arrival window. The first N128 profile tests that limitation. Additional profiled starts require the joint first-start P/D/credit screen. Actual D borrowing remains a separate unperformed gate.'))
    restore_if_idle()


if __name__=='__main__':main()
