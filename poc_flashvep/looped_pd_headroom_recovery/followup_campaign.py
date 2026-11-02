"""Qualify aggregated serving and prepared-D diagnostics after primary screens."""
import copy
import subprocess
from .common import *
from .prepare import put
from .initial_campaign import run
from .primary_campaign import request_rows,subset_parity
from .search_plan import make as primary_plan


def plan():
    fallback=ROOT/'state/followup_plan_v4.json'
    if fallback.exists():return json.loads(fallback.read_text())
    latest=ROOT/'state/followup_plan_v3.json'
    if latest.exists():return json.loads(latest.read_text())
    revised=ROOT/'state/followup_plan_v2.json'
    if revised.exists():return json.loads(revised.read_text())
    destination=ROOT/'state/followup_plan.json'
    if destination.exists():return json.loads(destination.read_text())
    primary=primary_plan()
    output=[]
    for record in primary['primaries']:
        model=record['model']
        source=json.loads((ROOT/'configs'/(record['pilots'][0]+'.json')).read_text())
        source.update(workload=f'L{source["length"]}_O{source["cap"]}',timeout=600)
        for key in ('capacity_reference','reference_capacity','rate_factor','rate_scope'):
            source.pop(key,None)
        item=dict(model=model,DP_gates=[],DP_pilots=[],prepared_D=[])
        for chunk in (512,2048):
            cfg=copy.deepcopy(source)
            cfg.update(name=f'aggregated_{model}_l{cfg["length"]}_c{chunk}_gate8',
                worker='native_mixed_worker',arm='DP2_NATIVE_MIXED',chunk=chunk,
                phase='correctness_DP2',wavefront=True,kv_blocks=22528 if model=='ouro' else 24576,
                capacity=8 if model=='ouro' else 4,slots=8 if model=='ouro' else 4,
                token_budget=8192 if model=='ouro' else 5120,graph_rows=8 if model=='ouro' else 4,
                verify_transfer=False,state_probe=False)
            item['DP_gates'].append(put(cfg,8))
            for rate in ([1.,2.,3.] if model=='ouro' else [.4,.8,1.2]):
                pilot=copy.deepcopy(cfg)
                pilot.update(name=f'aggregated_{model}_l{cfg["length"]}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    phase='aggregated_QPS_pilot')
                item['DP_pilots'].append(put(pilot,24,rate))
        for batch in (1,4 if model=='ouro' else 3,8 if model=='ouro' else 5):
            for pattern in ('all_ready','staggered'):
                cfg=copy.deepcopy(source)
                rate=2. if model=='ouro' else .8
                cfg.update(name=f'prepared_D_{model}_l{cfg["length"]}_b{batch}_{pattern}',
                    phase='prepared_D_diagnostic',worker='prepared_d_worker',prepare_D_only=True,
                    prepared_release_P_on_each_import=True,prepared_D_arrival_offsets=[0. if pattern=='all_ready' else i/rate for i in range(batch)],
                    D_ready_pattern=pattern,verify_transfer=False,state_probe=False)
                item['prepared_D'].append(put(cfg,batch))
        # The two measured profiles use matching clean inputs and arrivals.
        # The final candidate is not selected with these diagnostic traces.
        item['profiles']=[]
        for role in ('prefill','decode'):
            clean=record['pilots'][2]  # third ascending rate, atomic mode
            cfg=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
            cfg.update(name=f'profile_{model}_l{cfg["length"]}_{role}_primary24',
                phase='bounded_stage_profiling',P_only_profiler=True,nsys=True,capture_role=role,
                capture_seconds=3.,capture_prewarmed=True,capture_until_close=True,profile_stop_at_exit=True,
                nsys_cuda_flush_ms=300000,pd_shutdown_timeout=300,profile_finalization_timeout=300,
                observe_descriptor_cost=role=='prefill',record_descriptor_replay=role=='prefill',
                matched_clean=clean,timeout=600)
            cfg['native_transport_limits']=dict(cfg.get('native_transport_limits',{}),shutdown_timeout=300)
            # Preserve the exact already-generated Poisson schedule.
            write(ROOT/'configs'/(cfg['name']+'.json'),cfg)
            item['profiles'].append(cfg['name'])
        output.append(item)
    result=dict(created=time.time(),models=output,
        scope='Two native Mixed chunk budgets receive the same three rate points. DP2 requires exact parity before expansion. Prepared D uses actual native imports, not sustainable-capacity claims.')
    write(destination,result)
    return result


def main():
    while True:
        path=ROOT/'state/primary_campaign.json'
        value=json.loads(path.read_text()) if path.exists() else {}
        if value.get('stage')=='PRIMARY_MATRIX_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    ledger=[]
    for item in plan()['models']:
        model=item['model']
        primary=primary_plan()
        record=next(r for r in primary['primaries'] if r['model']==model)
        reference=record['gate'][0]
        if not (ROOT/'raw'/reference/'recovery_gate.json').exists():continue
        if not json.loads((ROOT/'raw'/reference/'recovery_gate.json').read_text())['passed']:continue
        write(ROOT/'state/followup_campaign.json',dict(stage='DP2_PARITY',model=model,time=time.time()))
        valid=[]
        for name in item['DP_gates']:
            okay=run(name) and subset_parity(reference,name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                okay=evidence.full_work(name)['passed']
            ledger.append(dict(name=name,passed=okay))
            if okay:valid.append(name)
        # Failed Mixed configurations do not block independent D diagnostics.
        for name in item['prepared_D']:
            okay=run(name) and subset_parity(reference,name)
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/followup_results.json',ledger)
        for name in item['DP_pilots']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            gate=next((g for g in valid if json.loads((ROOT/'configs'/(g+'.json')).read_text())['chunk']==cfg['chunk']),None)
            if gate is None:continue
            okay=run(name) and subset_parity(gate,name)
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/followup_results.json',ledger)
        for name in item['profiles']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            reference=ROOT/'raw'/cfg['matched_clean']/'dispatcher_result.json'
            deadline=time.monotonic()+1800
            while not reference.exists() and time.monotonic()<deadline:
                write(ROOT/'state/followup_campaign.json',dict(stage='WAIT_MATCHED_CLEAN',model=model,name=name,reference=cfg['matched_clean'],time=time.time()))
                time.sleep(5)
            if not reference.exists() or not json.loads(reference.read_text()).get('complete'):
                ledger.append(dict(name=name,passed=False,status='NOT_RUN_NO_COMPLETE_MATCHED_CLEAN'))
                continue
            okay=run(name)
            if okay:
                cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                if cfg.get('cuda_only_profile'):
                    from .cuda_only_profile import analyze
                    for role in cfg.get('capture_roles',['prefill']):
                        proof=analyze(name,cfg['matched_clean'],role)
                        okay=okay and proof.get('eligible_for_matched_clean',False)
                else:
                    from .profile_analysis import analyze
                    analyze(name,cfg['matched_clean'])
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/followup_results.json',ledger)
    write(ROOT/'state/followup_campaign.json',dict(stage='FOLLOWUP_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger))
    subprocess.run([str(PYTHON),'-m','poc_flashvep.looped_pd_headroom_recovery.analyze'],cwd=REPO,check=True)
    restore_if_idle()


if __name__=='__main__':main()
