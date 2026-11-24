"""Verify raw contracts and keep expected symptom failures separate from tests."""
import argparse
import itertools
import json
import subprocess
import time
import numpy as np
from .common import ROOT,SOURCE,REPO,PYTHON,START,initialize,preserve,write,digest
from .registry import historical_runs
from .reproduction import reconstruct


def historical():
    initialize();preserve()
    from ..dense_ar_longbench400_pd import independent_metric_audit
    previous=independent_metric_audit.ROOT;rows=[];source_hashes={}
    try:
        for item in historical_runs():
            independent_metric_audit.ROOT=item['root'];result=independent_metric_audit.one(item['name'])
            folder=item['root']/'raw'/item['name'];cfg,metric,requests=reconstruct(folder)
            cache=item['root']/'analysis'/(item['name']+'_metrics.json');cached=json.loads(cache.read_text())
            old={r['prompt_id']:r for r in cached['per_request']}
            for row in requests:
                observed=old[row['prompt_id']]
                for left,right in (('TTFT_PASS','TTFT_pass'),('ITL_PASS','every_ITL_pass'),('TPOT_PASS','TPOT_pass'),('JOINT_PASS','SLO_pass')):
                    assert row[left]==observed[right],(item['name'],row['prompt_id'],left)
            source_hashes[str(cache.relative_to(REPO))]=digest(cache)
            rows.append(dict(family=item['family'],label=item['label'],start=item['start'],**result,
                request_gate_booleans_verified=len(requests)))
    finally:independent_metric_audit.ROOT=previous
    assert all(digest(REPO/path)==sha for path,sha in source_hashes.items())
    value=dict(time=time.time(),passed=True,results=rows,cached_metrics_SHA256=source_hashes,
        scope='Read-only independent reconstruction of25 frozen client metrics and all four request predicates. '
            'Zero-output completion is invalid in the frozen implementation. One-output ITL is vacuously passing and TPOT=None passes. '
            'Historical raw and caches are not rewritten. This is not a new GPU run.')
    write(ROOT/'analysis/historical_independent_metric_audit.json',value);preserve()
    print('HISTORICAL_INDEPENDENT_METRIC_AUDIT_PASS',len(rows),flush=True);return value


def cpu():
    initialize();preserve();results=[]
    for args,expected in ((['-m','unittest','poc_flashvep.dense_ar_native_ttft_slo.test_metrics'],0),
            (['-m','poc_flashvep.dense_ar_native_ttft_slo.reproduction'],0),
            (['-m','poc_flashvep.dense_ar_native_ttft_slo.reproduction','--symptom-test','--minimal'],1)):
        command=[str(PYTHON),'-B',*args];value=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
        passed=value.returncode==expected
        if expected:passed &= 'REPRODUCED_SYMPTOM:' in value.stderr
        results.append(dict(command=command,expected_exit_code=expected,returncode=value.returncode,
            passed=passed,stdout=value.stdout,stderr=value.stderr))
    files=[]
    for path in sorted(SOURCE.glob('*.py')):
        compile(path.read_bytes(),str(path),'exec');files.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
    value=dict(time=time.time(),passed=all(row['passed'] for row in results),commands=results,source_files=files,
        scope='The minimized symptom command is expected red. It verifies the reported regression rather than correctness. '
            'Metric regression tests and raw reproduction must be green. Syntax compilation is not GPU correctness or scientific root-cause confirmation.')
    write(ROOT/'analysis/cpu_regression_audit.json',value);preserve();assert value['passed']
    print('CPU_REGRESSION_PASS',len(files),'source files',flush=True);return value


def arrivals():
    initialize();qual=json.loads((ROOT/'configs/root_cause_policy_qualification_v1.json').read_text());rows=[]
    items=list(qual['items'])
    common0=ROOT/'configs/native_common0_reference_addendum_v1.json'
    if common0.exists():items.append(json.loads(common0.read_text())['item'])
    for item in items:
        for stage in ('Natural32','Natural400'):
            path=ROOT/'configs'/(item['steps'][stage]+'.json');cfg=json.loads(path.read_text())
            expected=np.r_[0.,np.cumsum(np.random.default_rng(2026112401).exponential(.5,cfg['count']-1))].tolist()
            assert cfg['arrival_offsets']==expected
            rows.append(dict(name=cfg['name'],config_SHA256=digest(path),model_seed=cfg['seed'],
                qualification_arrival_generator_seed=2026112401,serial=False,
                scope='Qualification uses a documented independent fixed arrival generator. Config seed is the model seed here. No qualification timing is a primary performance start.'))
    path=ROOT/'jobs/main400_native_relative_policy_confirmation_v2.json'
    if path.exists():
        plan=json.loads(path.read_text())
        for block in plan['blocks']:
            old=REPO/block['historical_trace_config'];frozen=json.loads(old.read_text())
            for name in block['arms'].values():
                cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                assert cfg['arrival_offsets']==frozen['arrival_offsets'] and cfg['seed']==frozen['seed']
                rows.append(dict(name=name,model_seed=cfg['seed'],performance_arrival_seed=frozen['seed'],
                    exact_historical_offsets_passed=True,source_trace_SHA256=digest(old)))
    path=ROOT/'jobs/heldout192_native_relative_confirmation_v1.json'
    if path.exists():
        for block in json.loads(path.read_text())['blocks']:
            old=REPO/block['historical_trace_config'];frozen=json.loads(old.read_text())
            for name in block['arms'].values():
                cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                assert cfg['arrival_offsets']==frozen['arrival_offsets'] and cfg['seed']==frozen['seed']
                assert cfg['replay_output_lengths']==frozen['replay_output_lengths']
                rows.append(dict(name=name,model_seed=cfg['seed'],performance_arrival_seed=frozen['seed'],
                    exact_historical_offsets_passed=True,exact_output_lengths_passed=True,source_trace_SHA256=digest(old)))
    value=dict(time=time.time(),passed=True,rows=rows,
        scope='Explicitly separate qualification arrival generation from model seed. Primary timing reuses the exact historical Main400 offsets and model/arrival seed. No immutable config or raw event is edited.')
    write(ROOT/'analysis/arrival_seed_and_trace_audit.json',value)
    print('ARRIVAL_TRACE_AUDIT_PASS',len(rows),flush=True);return value


def datasets():
    from .runtime import copy_data
    initialize();names=('longbench_balanced400_v2.json','balanced400_v2_heldout.json',
        'balanced400_v2_discovery.json','qualification32.json','compiler_control8_original_v1.json')
    values={};rows=[]
    for name in names:
        path=copy_data(name);data=json.loads(path.read_text())
        ids={row['prompt_id'] for row in data};assert len(ids)==len(data)
        contexts={row['context_SHA256'] for row in data}
        values[name]=dict(ids=ids,contexts=contexts)
        rows.append(dict(dataset=name,SHA256=digest(path),requests=len(data),unique_request_ids=len(ids),
            unique_context_hashes=len(contexts),input_tokens=sum(row['input_tokens'] for row in data)))
    main,held,discovery=(values[name] for name in names[:3])
    assert len(main['ids'])==400 and len(held['ids'])==192 and len(discovery['ids'])==208
    assert main['ids']==held['ids']|discovery['ids'] and not held['ids']&discovery['ids']
    assert not held['contexts']&discovery['contexts']
    intersections=[dict(left=a,right=b,request_overlap=len(values[a]['ids']&values[b]['ids']),
        context_overlap=len(values[a]['contexts']&values[b]['contexts'])) for a,b in itertools.combinations(names,2)]
    value=dict(time=time.time(),passed=True,datasets=rows,intersections=intersections,
        scope='Every copied dataset matches its preserved parent byte for byte. Main400 is the union of Heldout192 and Discovery208. '
            'Heldout/Discovery contexts are disjoint. Correctness and serial calibration overlap are explicitly counted. '
            'No prompt,sampling,exclusion,truncation or natural-length contract is changed by this campaign.')
    write(ROOT/'analysis/dataset_identity_and_overlap_audit.json',value);preserve()
    print('DATASET_IDENTITY_AUDIT_PASS',intersections,flush=True);return value


def inventory(final=False):
    from .runtime import bind
    bind()
    from ..dense_ar_p_critical_path.verification import inventory as audited_inventory
    return audited_inventory(final)


def metrics(final=False):
    from .runtime import bind
    bind()
    from ..dense_ar_p_critical_path.verification import metrics as audited_metrics
    return audited_metrics(final)


def budget(final=False):
    preserve()
    from ..dense_ar_p_critical_path.verification import interval_union_seconds
    rows=[];pending=[];intervals={gpu:[] for gpu in range(8)}
    for folder in sorted((ROOT/'raw').iterdir()):
        command=folder/'command.json';receipt=folder/'dispatcher_result.json'
        if not command.exists():continue
        if not receipt.exists():pending.append(folder.name);continue
        cfg=json.loads(command.read_text());result=json.loads(receipt.read_text())
        begin=cfg['start_epoch'];duration=result['elapsed_seconds'];devices=cfg['GPUs']
        assert duration>=0 and set(devices)<=set(range(8))
        for gpu in devices:intervals[gpu].append((begin,begin+duration))
        rows.append(dict(name=folder.name,GPUs=devices,start_epoch=begin,elapsed_seconds=duration,
            reserved_GPU_process_hours=len(devices)*duration/3600,complete=result['complete'],
            command_SHA256=digest(command),receipt_SHA256=digest(receipt)))
    frozen=json.loads((ROOT/'state/budget.json').read_text())
    latest=max((r['start_epoch']+r['elapsed_seconds'] for r in rows),default=START)
    union={str(gpu):interval_union_seconds(value)/3600 for gpu,value in intervals.items()}
    value=dict(time=time.time(),passed=(not final or not pending) and latest<=frozen['deadline_epoch'],final=final,
        rows=rows,pending=pending,approved_maximum_wall_hours=frozen['maximum_hours'],approved_GPU_count=8,
        observed_dispatch_span_hours=(latest-START)/3600,
        sum_reserved_GPU_process_hours=sum(r['reserved_GPU_process_hours'] for r in rows),
        reserved_dispatch_interval_union_GPU_hours=sum(union.values()),per_GPU_union_hours=union,
        scope='Owned dispatcher intervals include startup,correctness,failed jobs and timing. '
            'The per-GPU union avoids double counting. Burner and CPU-only time are excluded. '
            'This is reserved process time,not GPU-exclusive compute or serving capacity.')
    write(ROOT/'analysis/gpu_budget_accounting.json',value)
    from .failure_analysis import save_csv
    save_csv(ROOT/'figures/gpu_budget_dispatches.csv',rows)
    if final:assert value['passed']
    print('GPU_BUDGET_AUDIT',len(rows),'dispatches',len(pending),'pending',flush=True);return value


def figures():
    from .figures import build
    from .intervention_figures import build as build_extra
    preserve();build();build_extra()
    def snapshot():
        rows=[]
        for name in ('figure_manifest.json','intervention_figure_manifest.json'):
            manifest=json.loads((ROOT/'analysis'/name).read_text())
            for row in manifest['figures']:
                for extension,sha in row['files'].items():
                    path=ROOT/'figures'/(row['stem']+'.'+extension)
                    assert digest(path)==sha
                    rows.append(dict(path=str(path.relative_to(REPO)),SHA256=sha))
        return sorted(rows,key=lambda row:row['path'])
    first=snapshot();build();build_extra();second=snapshot()
    result=dict(time=time.time(),passed=first==second,files=second,repeated_file_count=len(second),
        scope='Regenerate every measured/diagnostic/offline figure from unchanged analysis tables twice. '
            'Require exact PNG,PDF and CSV bytes. Source-date epoch is fixed and PDF date metadata is omitted. '
            'Byte identity does not replace scientific scope checks or visual inspection.')
    write(ROOT/'analysis/figure_regeneration_audit.json',result);preserve();assert result['passed']
    print('FIGURE_BYTE_REGENERATION_PASS',len(second),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('historical','cpu','arrivals','datasets','inventory','metrics','budget','figures'))
    parser.add_argument('--final',action='store_true');args=parser.parse_args()
    globals()[args.action](args.final) if args.action in ('inventory','metrics','budget') else globals()[args.action]()
