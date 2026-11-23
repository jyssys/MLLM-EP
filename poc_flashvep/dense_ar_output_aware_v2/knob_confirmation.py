"""Confirm compiler,credit and admission interventions with fixed output work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,dispatch_allowed,preserve
from .c3_campaign import qualified_registry,performance,launch,wait_file
from .c3_policy import policy_registry
from .statistics import ratio,safety
from .c0_analysis import csv_save

def prepare():
    registry=policy_registry();original=qualified_registry();prefix='Hybrid'
    mapping=dict(PCompiled=f'{prefix}_CompiledPrefix_Cap2_Credit8',PEager=f'{prefix}_EagerPrefix_Cap2_Credit8',
        Credit16=f'{prefix}_Cap2_Credit16',Cap1=f'{prefix}_Cap1_Credit8',Cap4=f'{prefix}_Cap4_Credit8',
        NativeCommon1='NativeCommon_Cap1_Credit0',NativeCommon2='NativeCommon_Cap2_Credit0',NativeCommon4='NativeCommon_Cap4_Credit0')
    missing=[k for k in mapping.values() if k not in registry]
    if missing:
        write(ROOT/'state/c3_knob_confirmation_complete_v2.json',dict(time=time.time(),complete=False,status='NOT_RUN',
            reason='Required intervention failed its composition gate',missing=missing));return None
    actions={k:registry[v] for k,v in mapping.items()};actions['NativeStrong']=original['NativeStrong']
    optional={label:f'{prefix}_{variant}_Cap2_Credit8' for label,variant in
        (('DCompiled','DCompiledInterval'),('DEager','DEagerInterval'),('DFused','FusedShort_DCompiledInterval'))}
    if optional['DFused'] not in registry and optional['DFused']+'_ContextFixV15' in registry:
        optional['DFused']+='_ContextFixV15'
    extra_comparisons=[]
    if all(optional[k] in registry for k in ('DCompiled','DEager')):
        actions.update({k:registry[optional[k]] for k in ('DCompiled','DEager')})
        extra_comparisons.append(['DCompiled','DEager'])
        if optional['DFused'] in registry:
            actions['DFused']=registry[optional['DFused']];extra_comparisons.append(['DFused','DCompiled'])
    grid=json.loads((ROOT/'configs/c3_common_qps_grid_v2.json').read_text());qps=min(grid['QPS_grid'],key=lambda q:abs(q-1.2*grid['reference_knee']))
    state=json.loads((ROOT/'state/official_compiled_c1_reference_complete_v2.json').read_text())
    lengths={r['prompt_id']:r['output_tokens'] for r in map(json.loads,(ROOT/'raw'/state['name']/'requests.jsonl').read_text().splitlines())}
    rows=json.loads((ROOT/'data/balanced400_v2_heldout.json').read_text());outputs=[lengths[r['prompt_id']] for r in rows]
    lanes=[[],[],[],[]];blocks=[]
    for start in range(3):
        arms={};seed=2026112311+start
        order=list(actions) if start%2 else list(reversed(actions))
        for label in order:
            name=f'V2_C3KnobReplay_q{qps}_s{start}_{label}'
            performance(actions[label],name,'balanced400_v2_heldout.json',[4,5],qps,seed,replay_lengths=outputs)
            lanes[2].append(name);arms[label]=name
        blocks.append(dict(start=start,seed=seed,qps=qps,arms=arms,pair=[4,5]))
    filename='c3_knob_independent_confirmation_v2.json'
    write(ROOT/'jobs'/filename,dict(time=time.time(),lanes=lanes,blocks=blocks,stop_on_failure=False,
        comparisons=[['PCompiled','PEager'],['Credit16','PCompiled'],['Cap4','Cap1']]+extra_comparisons,
        scope='Three fresh paired starts on the same pair and heldout192 originals. '
            'Each request gets the same independent compiled-C1 output length in every arm. This is controlled replay,not natural serving. '
            'Compiler on/off uses the same source5 and changes only the P compiler flag. D execution remains unchanged. '
            'Credit8/16 uses the same cut,cap2 and compiled prefix. Cap1/4 uses the same cut,credit8 and compiler. '
            'If both D interval variants pass their composition gates,include their same-source compiler-only intervention. '
            'If its fused-short combination also passes,include that one-flag intervention. '
            'Future EOS is experimental output control,not an online feature. Preserve failed safety and negative latency outcomes.'))
    return filename

def analyze():
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.service_accounting import analyze as service
    plan=json.loads((ROOT/'jobs/c3_knob_independent_confirmation_v2.json').read_text());values={};stages={};rows=[]
    for b in plan['blocks']:
        values[b['start']]={};stages[b['start']]={}
        for label,name in b['arms'].items():
            values[b['start']][label]=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
            stages[b['start']][label]=service(name)['means']
    for left,right in plan['comparisons']:
        aa=[values[k][left] for k in range(3)];bb=[values[k][right] for k in range(3)]
        assert all(a['output_tokens']==b['output_tokens'] for a,b in zip(aa,bb))
        gates={}
        for label,arm in ((left,aa),(right,bb)):
            cap=1 if label=='Cap1' else 4 if label=='Cap4' else 2
            gates[label]=safety(arm,{'NativeStrong':[values[k]['NativeStrong'] for k in range(3)],
                'NativeCommon':[values[k]['NativeCommon'+str(cap)] for k in range(3)]})
        rows.append(dict(left=left,right=right,paired_starts=3,output_work_exact=True,evidence='CONTROLLED_REPLAY',
            ratios={f:ratio([r[f] for r in aa],[r[f] for r in bb]) for f in
                ('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second')},
            decode_gates=gates,per_start_additive_mean_host_delta_ms=[{f:stages[k][left][f]-stages[k][right][f]
                for f in stages[k][left]} for k in range(3)],
            scope='A controlled runtime intervention identifies the total effect of its knob. '
                'Batch geometry and queueing can mediate that effect. Host enclosures are not GPU-exclusive operator costs. '
                'The additive stages explain mean TTFT,not TTFT p95. This is not a natural-EOS serving-speedup claim.'))
    write(ROOT/'analysis/c3_knob_independent_confirmation_v2.json',dict(time=time.time(),rows=rows))
    csv_save(ROOT/'figures/c3_knob_independent_confirmation_v2.csv',rows)
    print('C3_PAIRED_KNOB_INTERVENTIONS',[(r['left'],r['right'],r['ratios']['TTFT_p95_ms']) for r in rows],flush=True)

def run():
    wait_file(ROOT/'state/c3_continuous_iso8_output_bridge_complete_v2.json')
    if (ROOT/'configs/c3_knob_early_dispatch_priority_v9.json').exists():
        # These interventions were fixed before policy timing. They do not use
        # the policy performance or final selection,so the old barrier is not
        # a scientific dependency. Keep their heldout results out of selection.
        for lane in (0,1):
            wait_file(ROOT/'state'/f'c3_policy_qualification_v2_lane{lane}_complete_v4.json')
            wait_file(ROOT/'state'/f'c3_fused_teacher_context_v15_lane{lane}_complete_v4.json')
    else:wait_file(ROOT/'state/c3_policy_tuning_complete_v2.json')
    name=prepare()
    if name is None:return
    launch(name);analyze();write(ROOT/'state/c3_knob_confirmation_complete_v2.json',dict(time=time.time(),complete=True))

def advance_after_qualification(waiter_pid):
    preserve();assert dispatch_allowed()
    assert not (ROOT/'jobs/c3_knob_independent_confirmation_v2.json').exists()
    assert (ROOT/'state/c3_continuous_iso8_output_bridge_complete_v2.json').exists()
    assert not (ROOT/'state/c3_policy_tuning_complete_v2.json').exists()
    for lane in (0,1):
        for prefix in ('c3_policy_qualification_v2','c3_fused_teacher_context_v15'):
            assert (ROOT/'state'/f'{prefix}_lane{lane}_complete_v4.json').exists()
    process=Path('/proc')/str(waiter_pid);assert process.stat().st_uid==os.getuid()
    command=(process/'cmdline').read_bytes().decode().split('\0')
    assert command[:4]==[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_output_aware_v2.knob_confirmation']
    assert not (process/'task'/str(waiter_pid)/'children').read_text().strip(),'Do not replace an active GPU controller'
    write(ROOT/'configs/c3_knob_early_dispatch_priority_v9.json',dict(time=time.time(),
        old_dependency='c3_policy_tuning_complete_v2.json',
        new_dependency='Completed strict8/natural32/400 composition proofs and the completed iso-work bridge.',
        pair=[4,5],dataset='balanced400_v2_heldout.json',settings_changed=False,
        heldout_results_enter_static_selection=False,
        reason='Use an idle approved pair while independent policy calibration continues on0/1 and2/3. '
            'Every intervention and its N3 output replay were already fixed. No policy-performance input is required. '
            'Preserve every method,source,QPS,arrival seed,pair,order,output contract and safety threshold. '
            'Do not inspect these heldout timing results before the static selection freezes.'))
    os.kill(waiter_pid,signal.SIGTERM)
    target=ROOT/'logs/knob_confirmation_controller_early_v9.log';assert not target.exists()
    with target.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_output_aware_v2.knob_confirmation'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/knob_cpu_waiter_handoff_v9.json',dict(time=time.time(),old_PID=waiter_pid,
        old_command=command,new_PID=child.pid,old_waiter_had_no_children=True,active_GPU_jobs_interrupted=False))
    preserve();print('EARLY_FIXED_KNOB_CONTROLLER',child.pid,flush=True)

if __name__=='__main__':run()
