"""Select bounded replication points from actual pilot evidence, not an oracle."""
import copy
import statistics
import subprocess
from .common import *
from .prepare import put
from .initial_campaign import run
from .primary_campaign import request_rows,subset_parity


def select():
    target=ROOT/'state/confirmation_plan.json'
    if target.exists():return json.loads(target.read_text())
    configure()
    from ..looped_pd_atomic_prefill.analyze import serving_one
    from .gates import one
    candidates=[]
    for path in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or cfg.get('phase') not in ('qps_pilot','aggregated_QPS_pilot'):continue
        terminal=ROOT/'raw'/cfg['name']/'dispatcher_result.json'
        if not terminal.exists() or not json.loads(terminal.read_text())['complete']:continue
        if cfg.get('measurement_epoch')!='WARM_GRAPH_DURABLE_V2':continue
        row=serving_one(cfg)[0]
        if row is None:continue
        if cfg['model']=='ouro' and (cfg['length'],cfg['cap']) not in ((6144,64),(8192,32)):continue
        if cfg['model']=='huginn' and (cfg['length'],cfg['cap']) not in ((2560,32),(3072,32)):continue
        gate=one(cfg) if not cfg['arm'].startswith('DP') else None
        candidates.append((cfg,row,gate))
    plans=[];selections=[]
    for model in ('ouro','huginn'):
        pd=[v for v in candidates if v[0]['model']==model and not v[0]['arm'].startswith('DP')]
        if not pd:
            selections.append(dict(model=model,status='NO_COMPLETE_PRIMARY_PD_PILOT'))
            continue
        # Favor valid D SLO/credit first. Then favor real D-clear P backlog
        # and actual P service. Missing kernels cannot earn a positive gate.
        def key(value):
            cfg,row,g=value
            return (bool(g['D_SLO_gate'] and g['D_credit_gate']),
                g['D_credit_refusal_sample_fraction'] is not None and g['D_credit_refusal_sample_fraction']<.1,
                (g['P_ready_lower_bound_half_growth'] or 0)>0,
                min(g.get('P_fraction_reference') or 0,1.1),
                -(row.get('token_ITL_ms_p99') or 1e9))
        chosen=max(pd,key=key)
        cfg,row,g=chosen
        paired=[v for v in candidates if v[0]['model']==model and v[0]['arm'].startswith('DP')
            and (v[0]['length'],v[0]['cap'])==(cfg['length'],cfg['cap'])
            and abs(v[0]['arrival_rate']-cfg['arrival_rate'])<1e-8]
        if not paired:
            # A fair new DP2 point can reuse a parity-qualified chunk setting
            # at the selected PD QPS. It receives the same confirmation inputs.
            paired=[v for v in candidates if v[0]['model']==model and v[0]['arm'].startswith('DP')
                and (v[0]['length'],v[0]['cap'])==(cfg['length'],cfg['cap'])]
        dp=max(paired,key=lambda v:(v[1].get('SLO_goodput') if isinstance(v[1].get('SLO_goodput'),(int,float)) else -1,
            v[1]['requests_per_s'])) if paired else None
        selection=dict(model=model,PD_pilot=cfg['name'],DP_pilot=dp[0]['name'] if dp else None,
            QPS=cfg['arrival_rate'],pilot_gates=g,
            status='STRICT_CANDIDATE' if g['P_compute_gate'] and g['D_SLO_gate'] and g['D_credit_gate'] else 'CAUSAL_NEGATIVE_OR_UNRESOLVED_REPLICATION',
            scope='A selected closest point is not a positive regime. Selection does not replace a missing kernel trace or a failed Decode tail gate.')
        selections.append(selection)
        original=json.loads((ROOT/'data'/cfg['dataset']).read_text())
        heldout=original[24:152]
        assert len(heldout)==128
        data=ROOT/'data'/f'confirmation_{model}_L{cfg["length"]}.json'
        if data.exists():assert json.loads(data.read_text())==heldout
        else:write(data,heldout)
        for start,seed,pair in [(1,2026110211,[0,1]),(2,2026110212,[4,5]),(3,2026110213,[2,3])]:
            item=dict(model=model,start=start,PD=None,DP=None)
            for role,source in [('PD',cfg),('DP',dp[0] if dp else None)]:
                if source is None:continue
                out=copy.deepcopy(source)
                out.update(name=f'confirm_{model}_primary_{role}_start{start}_n128',
                    phase='final_repeat',independent_start=start,gpus=pair,arrival_seed=seed,
                    timeout=1800,verify_transfer=False,state_probe=False,observe_descriptor_cost=False,
                    record_descriptor_replay=False,nsys=False,P_only_profiler=False)
                # Same prompt-specific generation seeds across paired policies
                # and starts. Arrival seeds differ across independent starts.
                out['seed']=2026110220
                item[role]=put(out,128,cfg['arrival_rate'],split='confirmation')
            plans.append(item)
    result=dict(created=time.time(),selection=selections,runs=plans,
        rule='Pilot selection is exploratory. Each final start uses 128 distinct new corpus records, excluded from the 24 pilot prompts. Three independent process starts cross GPU pairs. Pair PD and Native Mixed DP2 at identical offered QPS and arrivals.',
        source='Actual completed primary and generic-control pilots. No future arrival or output information selects a per-request action.')
    write(target,result)
    return result


def main():
    while True:
        path=ROOT/'state/metadata_controls.json'
        value=json.loads(path.read_text()) if path.exists() else {}
        ladder=ROOT/'state/adaptive_ladder.json'
        h=ROOT/'state/huginn_primary_recovery.json'
        ladder_done=ladder.exists() and json.loads(ladder.read_text()).get('stage')=='ADAPTIVE_LADDER_COMPLETE_REVIEW_REQUIRED'
        h_done=h.exists() and json.loads(h.read_text()).get('stage')=='HUGINN_PRIMARY_RECOVERED_REVIEW_REQUIRED'
        epoch=ROOT/'state/warm_campaign.json'
        epoch_done=epoch.exists() and json.loads(epoch.read_text()).get('stage')=='WARM_EPOCH_COMPLETE_REVIEW_REQUIRED'
        longer=ROOT/'state/warm_secondary.json'
        longer_done=longer.exists() and json.loads(longer.read_text()).get('stage')=='WARM_SECONDARY_COMPLETE_REVIEW_REQUIRED'
        pressure=ROOT/'state/high_rate_extension.json'
        pressure_done=pressure.exists() and json.loads(pressure.read_text()).get('stage')=='HIGH_RATE_COMPLETE_REVIEW_REQUIRED'
        graph_limit=ROOT/'state/graph_safe_decode_control.json'
        graph_limit_done=graph_limit.exists() and json.loads(graph_limit.read_text()).get('stage')=='GRAPH_LIMIT_CONTROL_COMPLETE_REVIEW_REQUIRED'
        if value.get('stage')=='GENERIC_CONTROLS_COMPLETE_REVIEW_REQUIRED' and ladder_done and h_done and epoch_done and longer_done and pressure_done and graph_limit_done:break
        time.sleep(5)
    plan=select()
    write(ROOT/'state/confirmation.json',dict(stage='CONFIRMATION_PLAN_READY',time=time.time()))
    while True:
        profile=ROOT/'state/warm_profile.json'
        if profile.exists() and json.loads(profile.read_text()).get('stage')=='WARM_PROFILE_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    ledger=[];blocked=set();previous_PD={}
    for item in plan['runs']:
        if item['model'] in blocked:continue
        reference=None
        for role in ('PD','DP') if item['start']%2 else ('DP','PD'):
            name=item[role]
            if name is None:continue
            write(ROOT/'state/confirmation.json',dict(stage='REPLICATION',name=name,time=time.time()))
            okay=run(name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                work=evidence.full_work(name)
                write(ROOT/'raw'/name/'confirmation_work.json',work)
                okay=work['passed']
                if role=='PD' and item['model'] in previous_PD:
                    okay=okay and subset_parity(previous_PD[item['model']],name)
                if role=='PD':previous_PD.setdefault(item['model'],name)
                if reference is not None:okay=okay and subset_parity(reference,name)
                reference=name
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/confirmation_results.json',ledger)
            if not okay:
                blocked.add(item['model'])
                write(ROOT/'state'/f'confirmation_blocked_{item["model"]}.json',dict(name=name,time=time.time(),status='REVIEW_REQUIRED'))
                # Continue the other model, not an unqualified method.
                break
    write(ROOT/'state/confirmation.json',dict(stage='CONFIRMATION_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger))
    restore_if_idle()


if __name__=='__main__':main()
