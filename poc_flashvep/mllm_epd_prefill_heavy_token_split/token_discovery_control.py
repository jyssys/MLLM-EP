"""Fixed discovery protocol for protected static fractions and idle fallback."""
import asyncio
import json
from pathlib import Path
import random
import time

from . import token_correctness_control as single
from .client import execute, load_replay_counts, resolve_manifest
from .common import ROOT, digest, experiment_deadline, preserve, write
from .frontend_supply_control import retire
from .ipc_ec_control import launch, ownership_empty, stop_proxy
from .metrics import summarize
from .native_common_control import set_common_ipc_policy
from .source_snapshot import snapshot
from .token_split import select_prefix

TAG = 'token_DISCOVERY_C34'
FRACTIONS = [.1, .2, .25, .3, .5]
URL = 'http://127.0.0.1:18847'


def closed(tag):
    for role,gpu in [('E',4),('P',5),('D',6)]:
        lease=json.loads((ROOT/'state'/f'server_{tag}_{role}{gpu}.json').read_text())
        assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()


def selected_rows():
    path=resolve_manifest(None)
    manifest=json.loads(path.read_text());ids=set(manifest['discovery_ids'])
    rows=[row for row in manifest['requests'] if row['request_id'] in ids]
    assert len(rows)==50 and ids.isdisjoint(manifest['heldout_ids'])
    random.Random(2026112404).shuffle(rows)
    return path,rows


async def measure(label,url,rows,output,mode,metadata):
    run=TAG+'_'+label
    result=await execute(rows,[url],run,2.0,2026112402,output,
        metadata=dict(evidence='DIAGNOSTIC_WARMUP' if mode=='warmup' else 'MEASURED_ONLINE runtime screening',
            scope='Discovery50 only. One initialization per family. No primary GO or confidence interval.',
            task_quality='PENDING',GPU_observer='OFF',generation=mode,**metadata))
    assert len(result)==len(rows) and all(row['status']=='MEASURED_ONLINE' for row in result)
    ownership_empty(run,'cuda_ipc')
    value=summarize(run)
    print('TOKEN_DISCOVERY_METRICS',label,{key:value[key] for key in
        ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput','gate_attainment']},flush=True)
    return value


def proxy_state(url):
    import urllib.request
    with urllib.request.urlopen(url+'/campaign_state',timeout=30) as stream:
        state=json.load(stream)
    assert state['decode_pending']==state['encoder_pending']==state['prefill_pending']==0
    if state.get('continuation_admission') is not None:
        gate=state['continuation_admission']
        assert gate['active_http_requests']==gate['admission_waiting']==0 and gate['prefill_owner'] is None
    return state


def run_native(rows,counts):
    names=[];proxy=None;results=[]
    tag=TAG+'_Native'
    try:
        names,proxy=launch(tag,'cuda_ipc',fixed_reduction=True,deterministic_reductions=True,owned_names=names)
        set_common_ipc_policy()
        for mode,output in [('warmup',counts),('natural',None),('replay',counts)]:
            result=asyncio.run(measure('Native_'+mode,URL,rows,output,mode,
                dict(method='Common-policy Native',no_token_adapter=True,source_credit='Uncapped',D_admission='Original')))
            if mode!='warmup':results.append(dict(method='Native',fraction=0.,mode=mode,metrics=result))
        write(ROOT/'state'/f'{tag}_proxy_drained.json',proxy_state(URL))
    finally:
        if proxy is not None:stop_proxy(proxy)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])
    return results


def runtime_eligible(candidate,baseline):
    return candidate['gate_attainment']['ITL_PASS']>=.95 and candidate['gate_attainment']['TPOT_PASS']>=.95 and \
        candidate['TTFT_p50_ms']/baseline['TTFT_p50_ms']<=1.05 and \
        candidate['ITL_p99_ms']/baseline['ITL_p99_ms']<=1.05 and \
        candidate['TPOT_p95_ms']/baseline['TPOT_p95_ms']<=1.05


def choose(results):
    baseline=next(row['metrics'] for row in results if row['method']=='Native' and row['mode']=='replay')
    candidates=[]
    for fraction in FRACTIONS:
        row=next(row for row in results if row['method']=='Static Token' and row['fraction']==fraction and row['mode']=='replay')
        item=dict(fraction=fraction,TTFT_p95_ms=row['metrics']['TTFT_p95_ms'],
            SLO_goodput=row['metrics']['SLO_goodput'],runtime_point_gate=runtime_eligible(row['metrics'],baseline))
        candidates.append(item)
    eligible=[row for row in candidates if row['runtime_point_gate']]
    chosen=min(eligible or candidates,key=lambda r:(r['TTFT_p95_ms'],-r['SLO_goodput'],r['fraction']))
    best_goodput=max(eligible or candidates,key=lambda r:(r['SLO_goodput'],-r['TTFT_p95_ms'],-r['fraction']))
    return dict(time=time.time(),fraction=chosen['fraction'],candidates=candidates,
        best_goodput_fraction=best_goodput['fraction'],
        runtime_point_gate=chosen['runtime_point_gate'],selection='Lowest replay TTFT p95, then higher goodput, then smaller fraction.',
        quality='PENDING',confidence_interval='NOT_AVAILABLE',
        evidence='Discovery-only point selection. Not a primary decode gate, safe-window predictor, or confirmed serving gain.')


def run_token(rows,counts,results):
    tag=TAG+'_Token';single.TAG=tag
    single.EC=Path('/dev/shm')/f'mllm_epd_20261124_{tag}'
    names=[];teacher=None;proxy=None
    pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    order=FRACTIONS.copy();random.Random(2026112504).shuffle(order)
    try:
        single.launch_models(names,fixed_reduction=True,deterministic_reductions=True)
        teacher,teacher_url=single.launch_proxy(tag+'_teacher_proxy','e_pd',18848,names)
        for fraction in order:
            assert time.time()<experiment_deadline()
            suffix=f'F{int(round(fraction*100)):02d}'
            prefix=select_prefix(pool[0]['total_input_tokens'],fraction)
            teacher_run=f'{TAG}_{suffix}_N1_TEACHER'
            token_run=f'{TAG}_{suffix}_N1_TOKEN'
            teacher_result=single.capture(pool[0],teacher_run,teacher_url,[(18841,'P')],prefix)
            proxy,url=single.launch_proxy(tag+'_'+suffix+'_proxy','epd',18849,names,
                suffix_fraction=fraction,token_admission='decode_idle')
            try:
                token_result=single.capture(pool[0],token_run,url,[(18841,'P'),(18842,'D')])
                single.compare(pool[0],prefix,teacher_run,token_run,teacher_result,token_result)
                for mode,output in [('warmup',counts),('natural',None),('replay',counts)]:
                    result=asyncio.run(measure(suffix+'_'+mode,url,rows,output,mode,dict(
                        method='Protected Static Token',suffix_fraction=fraction,D_admission='Zero prior D HTTP streams',
                        additional_active_decode_insertion=False,numerical_policy='Common compiled',
                        baseline_scope='Static fractions compete under the same conservative safety contract. '
                            'This is not unrestricted active-decode insertion.')))
                    if mode!='warmup':results.append(dict(method='Static Token',fraction=fraction,mode=mode,metrics=result))
                write(ROOT/'state'/f'{tag}_{suffix}_proxy_drained.json',proxy_state(url))
            finally:
                stop_proxy(proxy);proxy=None
        selection=choose(results)
        choice_path=ROOT/'configs/token_DISCOVERY_C34_frozen_fraction_v0.json'
        assert not choice_path.exists();write(choice_path,selection)
        proxy,url=single.launch_proxy(tag+'_idle_proxy','epd',18849,names,
            suffix_fraction=selection['fraction'],token_admission='decode_idle',token_policy='idle_fallback')
        for mode,output in [('warmup',counts),('natural',None),('replay',counts)]:
            result=asyncio.run(measure('Idle_'+mode,url,rows,output,mode,dict(
                method='Idle-fallback Token',suffix_fraction=selection['fraction'],
                decision='Before encode, require P HTTP outstanding>0 and D HTTP outstanding=0. '
                    'Otherwise Native. Recheck zero D streams before actual suffix admission.',
                future_information='None',numeric_policy='Common compiled')))
            if mode!='warmup':results.append(dict(method='Idle-fallback Token',fraction=selection['fraction'],mode=mode,metrics=result))
        write(ROOT/'state'/f'{tag}_idle_proxy_drained.json',proxy_state(url))
        return selection
    finally:
        if proxy is not None:stop_proxy(proxy)
        if teacher is not None:stop_proxy(teacher)
        if names:
            retire([name for name in names if '_E4' not in name]);retire([name for name in names if '_E4' in name])


def main():
    preserve();assert time.time()<experiment_deadline()
    prior=ROOT/'analysis/token_NATURAL_C33_summary.json'
    proof=json.loads(prior.read_text());assert proof['passed']
    assert all(all(row['strict_tokens'] for row in cohort['full_prompt_comparisons']) for cohort in proof['cohorts'])
    closed('token_NATURAL_C33')
    manifest,rows=selected_rows();counts=load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    archive=snapshot(TAG)
    order=FRACTIONS.copy();random.Random(2026112504).shuffle(order)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_DISCOVERY_PERFORMANCE',source_archive_SHA256=archive['archive_SHA256'],
        manifest_SHA256=digest(manifest),prior_correctness_SHA256=digest(prior),GPU_group=[4,5,6],QPS=2.0,
        discovery_ids=[row['request_id'] for row in rows],heldout_touched=False,fraction_order=order,
        policy='Original continuous engine scheduler, common compiled P/D. Candidate extra prefill requires no prior D HTTP streams.',
        baseline='Common Native has no token adapter or new D admission gate. Stock NativeStrong N3 remains unchanged.',
        output_modes=['Natural EOS cap2048','Per-request equal-output replay'],one_start_per_family=True,
        select='Replay: retain runtime point-gate eligible candidates, minimize TTFT p95, then maximize goodput, then smaller fraction.',
        runtime_point_gate='All-ITL and TPOT request pass>=95%; TTFTp50,ITLp99 and TPOTp95 point ratio<=1.05 vs Native.',
        failed_gate_selection='If none pass, select lowest replay TTFT for diagnosis. Do not label it a serving winner.',
        quality='PENDING; raw-score floors and missing official MMLongDoc extraction prevent primary quality claims.',
        timing='No tensor capture, profiler or other GPU observer during measured cohorts. Warmups remain separate.',
        no_active_decode_extra_prefill=True,no_decode_safe_window_predictor=True))
    results=[]
    try:
        results.extend(run_native(rows,counts))
        write(ROOT/'analysis'/f'{TAG}_partial_results.json',dict(results=results,status='NATIVE_COMPLETE'))
        choice=run_token(rows,counts,results)
        write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE',results=results,selection=choice,
            task_quality='PENDING',evidence='MEASURED_ONLINE runtime discovery',independent_starts_per_family=1,
            not_a_confirmed_speedup=True,no_heldout_results_used=True))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_cells=len(results)))
        write(ROOT/'analysis'/f'{TAG}_partial_results_on_failure.json',dict(results=results))
        raise
    preserve()


if __name__=='__main__':main()
