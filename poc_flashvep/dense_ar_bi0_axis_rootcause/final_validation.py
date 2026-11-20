"""Check preserved inputs, installed sources and publication without GPU work."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from .common import ROOT,REPO,SOURCE,PREVIOUS_SOURCE,BRANCH,PYTHON,digest,preserve,resources,write
from .publication import DOCUMENTS


def tests(folder):
    command=[str(PYTHON),'-m','unittest','discover','-s',str(folder.relative_to(REPO)),'-t','.','-p','test_*.py']
    result=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)


def wheel():
    distribution=importlib.metadata.distribution('vllm');checked=0;mismatches=[]
    for entry in distribution.files or []:
        if not str(entry).startswith('vllm/') or not str(entry).endswith('.py') or not entry.hash:continue
        path=distribution.locate_file(entry)
        actual=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
        if entry.hash.mode!='sha256' or actual!=entry.hash.value:mismatches.append(str(entry))
        checked+=1
    assert checked>=2000 and not mismatches,(checked,mismatches)
    return dict(checked_python_sources=checked,mismatches=mismatches,version=distribution.version,
        scope='Installed Python sources versus wheel RECORD. No claim about every upstream build input or GPU binary.')


def links():
    results=[]
    paths=[REPO/'poc_flashvep/reports'/name for name in DOCUMENTS]
    paths.append(ROOT/'figures/README.md')
    for path in paths:
        text=path.read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            actual=(path.parent/target.split('#')[0]).resolve()
            results.append(dict(document=str(path.relative_to(REPO)),target=target,exists=actual.exists()))
    assert all(r['exists'] for r in results),[r for r in results if not r['exists']]
    return results


def main_contracts(stats):
    names=set()
    for row in stats['comparisons']:
        for values in row['absolute_start_means'].values():names.update(r['name'] for r in values['per_start'])
    rows=[]
    for name in sorted(names):
        folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
        assert cfg['gpus']==[0,1] and cfg['source_credit_limit']==4
        assert cfg['dataset_SHA256']=='8e58b2f75bd0c0ef274d6e53d9b4044e6aecf3a09a787133ee128b21f702e6d6'
        assert cfg['batch_invariant'] is False
        geometry=json.loads((ROOT/'analysis'/(name+'_timeline_analysis.json')).read_text())['geometry']
        for role in ('prefill','decode'):
            runtime=json.loads((folder/f'actual_runtime_{role}.json').read_text())
            expected_UUID={'prefill':'f217c8a0-1142-20f4-d84b-af29f3a47a0d','decode':'a77f3471-67d4-20b0-9fab-e502d4de5adb'}[role]
            assert runtime['uuid'].removeprefix('GPU-')==expected_UUID
            assert runtime['scheduler_B']==8192 and runtime['scheduler_S']==1024
            assert runtime['model_layers']==36 and runtime['parameter_dtypes']==['torch.bfloat16']
            assert runtime['KV_block_tokens']==128 and runtime['KV_bytes']<=42*1024**3
            assert runtime['batch_invariant'] is False
            assert runtime['compile_mode']=='0' and runtime['cudagraph_mode']=='FULL_DECODE_ONLY'
            assert runtime['options']['model'].endswith('b968826d9c46dd6066d109eabc6255188de91218')
            assert runtime['chunked_prefill'] and runtime['options']['enable_prefix_caching'] is False
            assert set(geometry[role]['actual_request_batch_histogram'])=={'1'}
        timeline=json.loads((ROOT/'analysis'/(name+'_axis_timeline.json')).read_text())
        assert timeline['all_prompt_work_accounted']
        assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
        rows.append(dict(name=name,passed=True,actual_geometry=geometry,
            stages_ms={key:value['mean'] for key,value in timeline['summary'].items()}))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true')
    parser.add_argument('--post-publication',action='store_true');args=parser.parse_args()
    preserve()
    source_hashes=json.loads((ROOT/'state/prior_source_hashes.json').read_text())
    assert all(digest(REPO/path)==value for path,value in source_hashes.items())
    reports=json.loads((ROOT/'state/prior_report_hashes.json').read_text())
    assert all(digest(REPO/'poc_flashvep/reports'/path)==value for path,value in reports.items())
    coordinator=SOURCE/'serving.py';expected='36a9c5bfeba591a5f4b5d202ee155af0bc5af06ced94469351b5fdabd79f3856'
    assert digest(coordinator)==expected,'Primary coordinator changed during confirmation'
    output=dict(time=time.time(),passed=True,old_sources_unchanged=len(source_hashes),
        old_reports_unchanged=len(reports),primary_coordinator_SHA256=expected,
        wheel_source_integrity=wheel(),local_report_links=links(),
        tests=[tests(SOURCE),tests(PREVIOUS_SOURCE)])
    compile_result=subprocess.run([str(PYTHON),'-m','compileall','-q',str(SOURCE)],capture_output=True,text=True)
    assert compile_result.returncode==0,compile_result.stderr
    output['compileall_passed']=True
    if args.final or args.post_publication:
        assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
        assert json.loads((ROOT/'state/final_GPU_resource_restoration.json').read_text())['passed']
        pending=[name for name in DOCUMENTS if 'IN_PROGRESS' in (REPO/'poc_flashvep/reports'/name).read_text()]
        assert not pending,pending
        stats=json.loads((ROOT/'analysis/full200_confirmation_statistics.json').read_text())
        assert {r['cut'] for r in stats['comparisons']}=={2,4,8,12}
        assert all(r['metrics']['TTFT_p95_ms']['starts']==3 for r in stats['comparisons'])
        output['four_cuts_three_start_confirmation']=True
        output['main_runtime_contract_checks']=main_contracts(stats)
    if args.post_publication:
        receipt=json.loads((ROOT/'state/publication_receipt.json').read_text())
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0]
        assert receipt['pushed'] and receipt['SHA']==receipt['Remote_SHA']==remote
        owner=resources();active=owner.processes();burners=owner.burners()
        assert {r['gpu'] for r in active}==set(range(8))
        assert all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active)
        output.update(Remote_SHA=remote,approved_burner_sessions=burners,active_GPU_processes=active,
            GPU_utilization=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu','--format=csv,noheader'],text=True),
            user_checkout_and_index_preserved=True)
        name='post_publication_final_check.json'
    else:name='final_CPU_validation.json' if args.final else 'preliminary_CPU_validation.json'
    preserve();write(ROOT/'state'/name,output)
    print(json.dumps(dict(passed=True,result=name,old_sources_unchanged=len(source_hashes),
        old_reports_unchanged=len(reports),installed_python_sources=output['wheel_source_integrity']['checked_python_sources'])))


if __name__=='__main__':main()
