"""Audit historical/current configuration differences without merging timing."""
import csv
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import textwrap
import time
import zipfile
from pathlib import Path
from .common import ROOT,V2,REPO,write,digest,preserve
from .failure_analysis import save_csv


def normalized(payload,name):
    if name=='poc_flashvep/dense_ar_longbench400_pd/common.py':
        return re.sub(rb"ROOT\s*=\s*REPO\s*/\s*'poc_flashvep/reports/[^']+'",b"ROOT=REPO/'OUTPUT_NAMESPACE'",payload)
    return payload


def scheduler_audit():
    package=Path(importlib.util.find_spec('vllm').origin).parent
    path=package/'v1/core/sched/scheduler.py';source=path.read_text();tree=ast.parse(source)
    owner=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Scheduler')
    node=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name=='schedule')
    segment=''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    function_sha=hashlib.sha256(segment.encode()).hexdigest()
    dedented=textwrap.dedent(segment)
    dedented_sha=hashlib.sha256(dedented.encode()).hexdigest()
    clip_lines=[]
    for assignment in ast.walk(ast.parse(dedented)):
        if not isinstance(assignment,ast.Assign) or len(assignment.targets)!=1:continue
        value=assignment.value
        if isinstance(assignment.targets[0],ast.Name) and assignment.targets[0].id=='num_new_tokens' and \
            isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='min' and \
            [ast.unparse(arg) for arg in value.args]==['num_new_tokens','token_budget']:
            clip_lines.append(assignment.lineno)
    clip_lines.sort();assert len(clip_lines)==2
    assert 'elif defer_prefills and num_computed_tokens < request.num_tokens - 1:' in segment
    observations=[]
    for family,parent in (('Historical_Main400',V2),('Current_campaign',ROOT)):
        for audit in sorted((parent/'raw').glob('*/D_compute_throttle.json')):
            if family=='Historical_Main400' and not audit.parent.name.startswith('V2_C3_Main400Natural_'):continue
            value=json.loads(audit.read_text())
            cfg=json.loads((audit.parent/'configuration.json').read_text())
            if family=='Historical_Main400' and cfg['qps']!=2.:continue
            hybrid=cfg.get('axis_kind')=='Hybrid' and cfg.get('partition_aligned_chunks')
            expected=dedented_sha if hybrid else function_sha
            transform=None;transform_pass=True
            if hybrid:
                transform_path=audit.parent/'partition_boundary_scheduler.json'
                transform=json.loads(transform_path.read_text())
                transform_pass=transform['original_source_SHA256']==dedented_sha and sorted(transform['changed_assignments'])==clip_lines
            observations.append(dict(family=family,name=audit.parent.name,
                observed_schedule_source_SHA256=value['scheduler_source_SHA256'],
                equals_current_source=value['scheduler_source_SHA256']==function_sha,
                matches_expected_source_form=value['scheduler_source_SHA256']==expected and transform_pass,
                expected_source_SHA256=expected,hybrid_AST_clip=transform,
                audit_SHA256=digest(audit)))
    result=dict(time=time.time(),passed=all(row['matches_expected_source_form'] for row in observations
        if row['family']=='Current_campaign'),installed_file=str(path),installed_file_SHA256=digest(path),
        schedule_function_SHA256=function_sha,dedented_schedule_source_SHA256=dedented_sha,
        hybrid_expected_clip_assignment_lines=clip_lines,schedule_lines=[node.lineno,node.end_lineno],observations=observations,
        predicates=dict(running_prefill_chunk='Skip when defer_prefills is true.',
            waiting_local_prefill='Block when more than one token remains.',
            waiting_remote_load='Async remote-KV allocation remains allowed before local compute.',
            native_last_query='A one-token remainder does not enter the multi-token prefill branch.',
            wrapper='The project wrapper disables the prefill_capacity_bound escape while requesting throttle_prefills.'),
        scope='Read-only source audit. Native/Layer/Token use the installed source form. '
            'Hybrid has a known pre-allocation AST clip at exactly two num_new_tokens sites. Its inspect source is the dedented baseline cached by that adapter. '
            'Validate the separate transform proof. Do not mistake the different indentation hash for a new runtime revision or claim the executed Hybrid code is identical. '
            'The one-token Native path and bulk continuations can have different eligibility under the same flag. '
            'Code eligibility is not a measured safe window,GPU idle or an independent latency contribution.')
    target=ROOT/'references/installed_scheduler_predicate_audit.json'
    if target.exists() and not json.loads(target.read_text())['passed']:
        preserved=ROOT/'references/scheduler_initial_reader_assumption_failure.json'
        if not preserved.exists():shutil.copyfile(target,preserved)
    write(target,result)
    assert result['passed'];return result


def analyze():
    preserve();dependency=scheduler_audit();plan=json.loads((ROOT/'jobs/main400_native_relative_policy_confirmation_v2.json').read_text())
    old=list(csv.DictReader((V2/'figures/c3_main_serving_v2_per_start.csv').open()))
    result=[];configuration_rows=[]
    labels=('NativeStrong','NativeCommon0','NativeCommon2','StockToken','Layer','Hybrid')
    bookkeeping={'name','dataset','dataset_SHA256','count','seed','arrival_offsets','gpus','scope',
        'source_capsule','source_capsule_SHA256','qualification_gate_v2','official_compiler_cache_regime'}
    for label in labels:
        history=next(row for row in old if row['label']==label and int(row['start'])==0 and float(row['QPS'])==2.)
        old_folder=V2/'raw'/history['name'];old_cfg=json.loads((old_folder/'configuration.json').read_text())
        name=plan['blocks'][0]['arms'][label];new_cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        for key in ('dataset_SHA256','seed','arrival_offsets','count','qps','ignore_eos','output_cap','batch_invariant',
                'token_budget','max_num_seqs'):assert old_cfg[key]==new_cfg[key],(label,key)
        changed={key:dict(old=old_cfg.get(key,'ABSENT'),current=new_cfg.get(key,'ABSENT'))
            for key in set(old_cfg)|set(new_cfg) if old_cfg.get(key,'ABSENT')!=new_cfg.get(key,'ABSENT')}
        runtime_changes={key:value for key,value in changed.items() if key not in bookkeeping}
        a=old_folder/'source.zip';b=ROOT/'references/source_capsules'/new_cfg['source_capsule']
        with zipfile.ZipFile(a) as left,zipfile.ZipFile(b) as right:
            before=set(left.namelist());after=set(right.namelist());common=before&after
            changed_blobs=[]
            for filename in sorted(common):
                x,y=left.read(filename),right.read(filename)
                if x==y:continue
                changed_blobs.append(dict(file=filename,old_SHA256=hashlib.sha256(x).hexdigest(),
                    current_SHA256=hashlib.sha256(y).hexdigest(),
                    same_after_ROOT_literal_normalization=normalized(x,filename)==normalized(y,filename)))
        result.append(dict(label=label,historical_name=history['name'],current_name=name,
            old_source_SHA256=digest(a),current_source_SHA256=digest(b),changed_configuration=changed,
            changed_runtime_configuration=runtime_changes,changed_common_source_blobs=changed_blobs,
            source_files_added=sorted(after-before),source_files_removed=sorted(before-after),
            scope='This bridge records source/config differences. It is not a one-factor causal performance comparison. '
                'Historical Main400 and fresh Main400 remain separate families. New one-factor policy contrasts are within the fresh family only.'))
        configuration_rows.extend(dict(method=label,field=key,**value) for key,value in runtime_changes.items())
    value=dict(time=time.time(),passed=True,comparisons=result,installed_scheduler_audit_SHA256=digest(ROOT/'references/installed_scheduler_predicate_audit.json'),
        scope='The Main400 prompt/arrival/output/B/S/BI0 contract is identical across the old/new bridge. '
            'Inherited coordinator hygiene,compiler paths,warm-cache regime and source snapshots can differ. '
            'Never attribute the whole old/new latency change to a single policy. Old raw reproduction is not a new GPU replication.')
    write(ROOT/'analysis/historical_current_runtime_bridge_audit.json',value)
    save_csv(ROOT/'figures/historical_current_runtime_configuration_bridge.csv',configuration_rows)
    preserve();return value


if __name__=='__main__':
    value=analyze();print('HISTORICAL_CURRENT_RUNTIME_BRIDGE_PASS',len(value['comparisons']),flush=True)
