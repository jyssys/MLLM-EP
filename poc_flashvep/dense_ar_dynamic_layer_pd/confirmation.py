"""Freeze discovery choices before any held-out GPU performance run."""
import fcntl,math
from .common import *
from .prepare import offsets
from .online_pilots import serving_config
from .run import execute
from .analyze import compare,metrics,records


def eligible_rows(prefixes,cap):
    rows=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        if not path.stem.startswith(tuple(prefixes)):continue
        c=json.loads(path.read_text())
        if c.get('cap')!=cap or c.get('count')!=64 or c.get('dataset')!='discovery.json':continue
        m=metrics(c['name'])
        if not m.get('complete'):continue
        gate=compare(f'coarse_O{cap}_N64_q4.777_native16_s1_v1',c['name'])
        if gate['passed']:rows.append(dict(name=c['name'],metrics=m,output_gate=gate))
    assert rows,(prefixes,cap)
    return rows


def select(rows,native):
    def safety(r):
        m=r['metrics']
        return max(m['ITL_p99_ms']/native['ITL_p99_ms'],m['TPOT_p95_ms']/native['TPOT_p95_ms'])
    safe=[r for r in rows if safety(r)<=1.05 and
        r['metrics']['SLO_regimes']['Balanced']['percentile_Decode_absolute_pass']]
    def quality(r):
        m=r['metrics'];s=m['SLO_regimes']['Balanced']
        return (-s['Request_SLO_goodput'],m['TTFT_p95_ms'],safety(r))
    if safe:return min(safe,key=quality),True
    return min(rows,key=lambda r:(safety(r),*quality(r))),False


def freeze_selection():
    target=ROOT/'configs/heldout_selection.json'
    if target.exists():return json.loads(target.read_text())
    chosen={};evidence={}
    for cap in (16,32,64):
        natives=eligible_rows(('lean_native_','frontier_Native_','online_NATIVE_'),cap)
        qualified=[r for r in natives if r['metrics']['SLO_regimes']['Balanced']['percentile_Decode_absolute_pass']]
        native_row=min(qualified or natives,key=lambda r:(
            -r['metrics']['SLO_regimes']['Balanced']['Request_SLO_goodput'],r['metrics']['TTFT_p95_ms']))
        native_name=native_row['name'];native=native_row['metrics']
        coloc=eligible_rows(('coarse_','strong_','frontier_coloc_','LA_strong_coloc_','layout_'),cap)
        coloc=[r for r in coloc if json.loads((ROOT/'configs'/(r['name']+'.json')).read_text())['arm']=='DP2']
        # Native-relative Decode isolation is not a tuning restriction on DP2.
        # DP2 is independently tuned for the frozen Balanced joint goodput.
        best_coloc=min(coloc,key=lambda r:(-r['metrics']['SLO_regimes']['Balanced']['Request_SLO_goodput'],
            r['metrics']['TTFT_p95_ms']))
        coloc_safe=max(best_coloc['metrics']['ITL_p99_ms']/native['ITL_p99_ms'],
            best_coloc['metrics']['TPOT_p95_ms']/native['TPOT_p95_ms'])<=1.05
        selected=dict(Native=native_name,Colocated=best_coloc['name'])
        details=dict(Native=native,Native_candidates=natives,Colocated_candidates=coloc,Colocated_point_safety=coloc_safe)
        if cap in (16,32):
            static=eligible_rows(('validation_',),cap)
            best_static,static_safe=select(static,native)
            dynamic=eligible_rows(('online_COST_','online_D8_COST_','predictor_COST_BATCH_','predictor_D16_COST_BATCH_'),cap)
            best_dynamic,dynamic_safe=select(dynamic,native)
            selected.update(Static=best_static['name'],COST=best_dynamic['name'])
            details.update(Static_candidates=static,Dynamic_candidates=dynamic,
                Static_point_safety=static_safe,Dynamic_point_safety=dynamic_safe)
        chosen[str(cap)]=selected;evidence[str(cap)]=details
    value=dict(time=time.time(),selected=chosen,evidence=evidence,
        SLO_SHA256=digest(ROOT/'configs/frozen_SLO_regimes.json'),
        discovery_prompt_SHA256=digest(ROOT/'data/discovery.json'),
        heldout_prompt_SHA256=digest(ROOT/'data/heldout.json'),
        arrival_seeds=[2026111341,2026111342,2026111343],
        primary_Output16_arrival_seeds=[2026111341,2026111342,2026111343,2026111344,2026111345],
        rate_by_output={'16':4.777100014491616,'32':6.015607425656108,'64':6.015607425656108},
        load_scope='Output16 confirms the primary policy condition. Output32/64 confirm the stronger performance trade-off observed in the high-load discovery pilots. Their runtime settings transfer from middle-load tuning without held-out retuning.',
        rule='For Static and Dynamic prefer relative Decode ratios at most1.05 plus the Balanced absolute gate. Then maximize Balanced request goodput and minimize TTFT p95. If none pass, minimize the worst relative Decode ratio. Tune Colocated independently by Balanced joint goodput and TTFT. Do not constrain its tuning to Native-relative Decode ratios.',
        confidence_unit='Independent server start. No request or token resampling substitutes for independent starts.',
        scope='All selections use discovery prompts and starts only. The held-out results cannot alter this selection.')
    write(target,value);write(ROOT/'state/heldout_selection_freeze_receipt.json',dict(time=time.time(),SHA256=digest(target)))
    files=('common.py','native.py','dynamic_native.py','extension.py','dynamic_extension.py','hidden_transport.py',
        'output_priming.py','async_ingress.py','worker.py','policy.py','credits.py','ack_retention.py',
        'token_protocol.py','boundaries.py','watch.py','activity.py')
    hashes={str((SOURCE/name).relative_to(REPO)):digest(SOURCE/name) for name in files}
    for package in ('dense_ar_layerwise_pd','looped_pd_atomic_prefill','looped_pd_abc'):
        hashes.update({str(p.relative_to(REPO)):digest(p) for p in (REPO/'poc_flashvep'/package).glob('*.py')})
    write(ROOT/'state/heldout_runtime_freeze.json',dict(time=time.time(),hashes=hashes,
        scope='All held-out and enclosure-control runs use the same frozen execution source. Analysis-only files can change without changing GPU execution.'))
    return value


def jobs(selection):
    result=[];arms_by_cap={}
    for cap in (16,32,64):
        templates={arm:json.loads((ROOT/'configs'/(name+'.json')).read_text())
            for arm,name in selection['selected'][str(cap)].items()}
        if cap==16:
            best=templates['COST'];budget=best.get('D_token_budget',32000)
            for mode in ('DEADLINE','IDLE','NATIVE'):
                templates[mode if mode!='NATIVE' else 'ControllerZero']=dict(best,policy=mode)
            source=ROOT/'raw'/best['name']/'actual_cut_commits.jsonl'
            commits=records(source)
            fraction=sum(len(r['requests']) for r in commits if r['offloaded_layers'])/64
            positive=[r['offloaded_layers'] for r in commits if r['offloaded_layers']]
            layers=max(set(positive),key=positive.count) if positive else 12
            templates['MatchedFraction']=serving_config('unused',policy='STATIC',D_token_budget=budget,
                static_layers=layers,migration_fraction=fraction,
                static_fraction_axis='REQUEST_ID_BLOCKS',static_request_block_size=4,
                fraction_scope='Frozen discovery COST request fraction. No held-out result sets this fraction.')
        arms_by_cap[str(cap)]=list(templates)
        seeds=selection['primary_Output16_arrival_seeds'] if cap==16 else selection['arrival_seeds']
        for j,seed in enumerate(seeds):
            order=list(templates)
            # Native first in the first start supplies a token reference.
            # Rotate the following starts to reduce fixed order effects.
            if j:order=order[j:]+order[:j]
            for arm in order:
                rate=selection['rate_by_output'][str(cap)]
                c=dict(templates[arm]);c.update(name=f'heldout_O{cap}_N128_s{seed}_{arm}_v1',
                    count=128,dataset='heldout.json',cap=cap,request_caps=[cap]*128,
                    arrival_seed=seed,arrival_offsets=offsets(128,rate,seed),
                    arrival_rate=rate,phase='FROZEN_SELECTION_HELDOUT_INDEPENDENT_CONFIRMATION',
                    evaluation_arm=arm,selection_SHA256=digest(ROOT/'configs/heldout_selection.json'),
                    native_ACK_hold=True,execution_events=False,timeout=900,wall_timeout=1300)
                result.append(c)
    write(ROOT/'state/heldout_plan.json',dict(time=time.time(),jobs=[save(c) for c in result],
        arms_by_cap=arms_by_cap,source_selection_SHA256=digest(ROOT/'configs/heldout_selection.json')))
    return result


def main():
    initialize();limit=time.monotonic()+18000
    waiting=('static_validation_results.json','wider_online_pilot_results.json',
        'colocated_layout_frontier_results.json','load_aware_colocated_results.json','load_refinement_results.json',
        'native_frontier_results.json','predictor_refinement_results.json','destination16_results.json',
        'fraction_control_results.json','native_prefill_frontier_results.json')
    while not all((ROOT/'analysis'/f).exists() for f in waiting):
        assert time.monotonic()<limit,'Validation did not complete. Do not select from partial results.'
        time.sleep(5)
    selected=freeze_selection();planned=jobs(selected);results=[]
    try:
        for c in planned:
            if not execute(c['name'],restore=False):return
            reference=f'heldout_O{c["cap"]}_N128_s{selected["arrival_seeds"][0]}_Native_v1'
            output=compare(reference,c['name'])
            row=dict(name=c['name'],arm=c['evaluation_arm'],cap=c['cap'],arrival_seed=c['arrival_seed'],
                output_gate=output,metrics=metrics(c['name']))
            results.append(row);write(ROOT/'state/heldout_progress.json',dict(results=results))
            print('HELDOUT',c['name'],output['passed'],row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
            if not output['passed']:return
        write(ROOT/'analysis/heldout_results.json',dict(results=results,selection_SHA256=digest(ROOT/'configs/heldout_selection.json')))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
