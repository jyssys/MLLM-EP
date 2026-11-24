"""Confirm predeclared policy contrasts with frozen equal-output requests."""
import argparse
import fcntl
import json
import subprocess
import time
import traceback
import numpy as np
from .common import ROOT,PRIOR,V2,REPO,PYTHON,write,digest,preserve,resources
from .runtime import bind,copy_data,dispatch_allowed

PLAN='heldout192_native_relative_confirmation_v1.json'
COMPLETE='heldout192_native_relative_complete_v1.json'
CORE_LABELS=('NativeStrong','NativeCommon0','NativeCommon2','StockToken','StockCap2','Layer','Hybrid')
ALL_LABELS=CORE_LABELS+('NativeRegOpen','LayerRegOpen','HybridRegOpen')


def budget_selection(now,ceiling,max_main,max_prime):
    full_estimate=30*(max_main+max_prime)*1.05
    core_estimate=21*(max_main+max_prime)*1.05
    selected=ALL_LABELS if now+full_estimate<ceiling else CORE_LABELS if now+core_estimate<ceiling else ()
    return dict(full_schedule_estimate_seconds=full_estimate,core_schedule_estimate_seconds=core_estimate,
        full_schedule_fits=now+full_estimate<ceiling,core_schedule_fits=now+core_estimate<ceiling,
        selected_labels=selected,selected_timing_jobs=3*len(selected))


def prepare():
    runner=bind();copy_data('balanced400_v2_heldout.json')
    data=json.loads((ROOT/'data/balanced400_v2_heldout.json').read_text())
    parent=json.loads((ROOT/'jobs/main400_native_relative_policy_confirmation_v2.json').read_text())
    labels=('NativeStrong','NativeCommon0','NativeCommon2','StockToken','StockCap2',
        'NativeRegOpen','Layer','LayerRegOpen','Hybrid','HybridRegOpen')
    length_source=PRIOR/'configs/PC_C3_EqualReplay_s0_OriginalLayer.json'
    lengths=json.loads(length_source.read_text())['replay_output_lengths']
    assert len(lengths)==len(data)==192 and all(0<n<=2048 for n in lengths)
    assert sum(lengths)==11208
    blocks=[]
    for start in range(3):
        source=V2/'configs'/f'V2_C3_HeldoutEqualReplay_q2.0_s{start}_NativeStrong.json'
        if not source.exists():
            # The exact historical name is read from its frozen per-start table.
            import csv
            match=[r for r in csv.DictReader((V2/'figures/c3_heldout_replay_v2_per_start.csv').open())
                if r['label']=='NativeStrong' and int(r['start'])==start and float(r['QPS'])==2.]
            assert len(match)==1
            source=V2/'configs'/(match[0]['name']+'.json')
        frozen=json.loads(source.read_text())
        assert frozen['dataset_SHA256']==digest(ROOT/'data/balanced400_v2_heldout.json')
        assert frozen['seed']==2026112281+start
        assert frozen['replay_output_lengths']==lengths
        order=list(labels) if start==0 else list(reversed(labels)) if start==1 else np.random.default_rng(2026112404).permutation(labels).tolist()
        arms={};primers={}
        for label in order:
            original=parent['blocks'][start]
            cfg=json.loads((ROOT/'configs'/(original['arms'][label]+'.json')).read_text())
            name=f'NS_Heldout192EqualReplay_s{start}_{label}'
            cfg.update(name=name,dataset='balanced400_v2_heldout.json',dataset_SHA256=frozen['dataset_SHA256'],
                count=192,seed=frozen['seed'],arrival_offsets=frozen['arrival_offsets'],serial=False,qps=2.,
                ignore_eos=True,output_cap=2048,replay_output_lengths=lengths,diagnostic_controlled_output_replay=True,
                scope='Fresh equal-output Heldout192 policy confirmation. All400 Main timing data remain a separate family. '
                    'These192 prompts also occur inside Main400. They are held out from Discovery208,not from Main measurement. '
                    'Runtime policies and the ten-arm selection were frozen before new Main performance results.')
            from ..dense_ar_output_aware_v2.work_contract import regenerate
            regenerate(cfg,data);runner.save(cfg);arms[label]=name
            prime=json.loads((ROOT/'configs'/(original['primers'][label]+'.json')).read_text())
            prime_name=f'NS_Heldout192Prime8_s{start}_{label}'
            prime.update(name=prime_name,seed=frozen['seed'])
            runner.save(prime);primers[label]=prime_name
        blocks.append(dict(start=start,pair=[0,1],seed=frozen['seed'],arms=arms,primers=primers,order=order,
            historical_trace_config=str(source.relative_to(REPO)),historical_trace_SHA256=digest(source)))
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),blocks=blocks,length_source=str(length_source.relative_to(REPO)),
        length_source_SHA256=digest(length_source),output_tokens_per_start=sum(lengths),
        scope='Preserve strong uncapped compiled Stock and exact Common0/Common2 Native references. '
            'Confirm registration-open controls and the one-factor Stock cap control. '
            'Do not repeat source-retirement Heldout controls already measured in the prior campaign. '
            'Run the entire proposed N3 schedule only if its empirical wall-time estimate fits before the dispatch ceiling. '
            'No selection uses the direction or significance of a new performance effect. '
            'All kernel/host,quality and decode failures remain distinct. Equal-output replay is not Natural EOS serving.'))
    log=ROOT/'logs/heldout192_native_relative_confirmation_v1.log'
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_native_ttft_slo.heldout_confirmation','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/heldout192_native_relative_controller_v1.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def replay_quality(name):
    """Use the frozen quality margin, while retaining the replay contract label."""
    from ..dense_ar_output_aware_v2.quality_gate import paired_quality
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    reference=json.loads((V2/'state/official_compiled_c1_reference_complete_v2.json').read_text())
    assert reference['passed']
    source=V2/'raw'/reference['name']/'requests.jsonl'
    refs={r['prompt_id']:r for r in map(json.loads,source.read_text().splitlines())}
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    actual=list(map(json.loads,(ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()))
    lengths=cfg['replay_output_lengths']
    assert len(actual)==len(data)==len(lengths)
    assert all(r['output_tokens']==lengths[r['sample_id']] for r in actual)
    result=paired_quality(actual,[refs[r['prompt_id']] for r in data],data,4000)
    contract=ROOT/'configs/c3_quality_contract_v2.json'
    margin=json.loads(contract.read_text())['paired_task_macro_margin_percentage_points']
    result.update(time=time.time(),name=name,passed=result['CI95_percentage_points'][0]>=-margin,
        margin_percentage_points=margin,contract_SHA256=digest(contract),reference_requests_SHA256=digest(source),
        scope='Task-quality diagnostic under equal-output replay. This is not the Natural EOS quality distribution. '
            'Every actual request length equals its frozen compiled-C1 natural length. No token output is overwritten.')
    write(ROOT/'analysis'/(name+'_quality_gate.json'),result)
    return {k:v for k,v in result.items() if k!='rows'}


def run():
    bind()
    from ..dense_ar_p_critical_path.stable_source import install
    from ..dense_ar_p_critical_path.warm_serving_followup import expected_roots,check_reuse
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from .qualification import state_audit
    runner=install();owner=resources();plan=json.loads((ROOT/'jobs'/PLAN).read_text());results=[]
    while not (ROOT/'state/main400_native_relative_complete_v2.json').exists():
        if not dispatch_allowed():break
        time.sleep(10)
    main_path=ROOT/'state/main400_native_relative_complete_v2.json'
    main=json.loads(main_path.read_text()) if main_path.exists() else dict(results=[])
    required=('NativeStrong','NativeCommon0','NativeCommon2','StockToken')
    baseline_ok=all(sum(row['label']==label and row['status']=='COMPLETE' and row.get('quality_passed',False)
        for row in main['results'])==3 for label in required)
    # Upper-bound the proposed follow-up with the observed Main durations.
    times=[]
    for row in main['results']:
        path=ROOT/'raw'/row['name']/'dispatcher_result.json'
        if path.exists():times.append(json.loads(path.read_text())['elapsed_seconds'])
    prime_times=[]
    for path in (ROOT/'raw').glob('NS_Main400Prime8_*/dispatcher_result.json'):
        prime_times.append(json.loads(path.read_text())['elapsed_seconds'])
    budget=json.loads((ROOT/'state/budget.json').read_text())
    selection=budget_selection(time.time(),budget['last_large_dispatch_epoch'],
        max(times,default=600),max(prime_times,default=300))
    selected=set(selection['selected_labels']);fits=bool(selected)
    write(ROOT/'state/heldout192_budget_decision_v1.json',dict(time=time.time(),baseline_ok=baseline_ok,
        **selection,amendment='D17_PREDECLARED_TIME_ONLY_CORE_FALLBACK',
        estimation_scope='Keep full30 if it fits. Otherwise keep21 jobs for the predeclared seven mandatory references and original-axis controls if they fit. '
            'Use maximum observed Main400 duration plus maximum actual Prime8 duration,with5% reserve. '
            'Core labels were fixed before any Heldout result and before completed Main N3. '
            'The estimate and label choice do not use effect signs or significance. The hard12h ceiling and90min closeout reserve do not change.'))
    with (ROOT/'state/pair_0_1.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            blocked=set()
            for block in plan['blocks']:
                for label in block['order']:
                    name=block['arms'][label]
                    if not baseline_ok or not fits or label not in selected or label in blocked or not dispatch_allowed():
                        reason='PREDECLARED_CORE_FALLBACK_BUDGET' if fits and label not in selected else 'BASELINE_OR_COMPLETE_N3_BUDGET_OR_PREVIOUS_FAILURE'
                        results.append(dict(name=name,label=label,start=block['start'],status='NOT_RUN',reason=reason));continue
                    value=dict(name=name,label=label,start=block['start'],status='FAILED')
                    try:
                        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                        proof=json.loads((ROOT/'state'/cfg['qualification_gate_v2']).read_text());assert proof['passed']
                        primer=block['primers'][label]
                        assert not (ROOT/'raw'/primer).exists() and not (ROOT/'raw'/name).exists()
                        owner.stop([0,1]);assert runner.execute(primer),primer
                        state=state_audit(primer);assert state['passed'];roots=expected_roots(primer)
                        owner.stop([0,1]);assert runner.execute(name),name
                        reuse=check_reuse(name,roots);quality=replay_quality(name);metric=summarize(name)
                        value.update(status='COMPLETE',quality_passed=quality['passed'],cache_reuse=reuse,primer_state=state,
                            metrics={key:metric[key] for key in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens',
                                'Joint_SLO_attainment','SLO_qualified_output_tokens_per_second')})
                        if not quality['passed']:blocked.add(label);value['evidence']='FAILED_QUALITY_GATE'
                    except Exception as error:
                        blocked.add(label);value.update(error=repr(error),traceback=traceback.format_exc())
                    results.append(value)
                    write(ROOT/'state/heldout192_native_relative_progress_v1.json',dict(time=time.time(),results=results))
                    print('NS_HELDOUT_CONFIRMATION',json.dumps({k:v for k,v in value.items() if k not in ('cache_reuse','primer_state','traceback')}),flush=True)
        finally:owner.restore([0,1]);preserve()
    write(ROOT/'state'/COMPLETE,dict(time=time.time(),complete=True,results=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    print(prepare() if args.action=='prepare' else run(),flush=True)
