"""Revise only an unlaunched attention-PoC protocol after a CPU API failure."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile
from .common import ROOT,REPO,SOURCE,PYTHON,write,digest,preserve
from .attention_poc import POC


def stop_pending_v1():
    record=POC/'state/controller_v1.json';old=json.loads(record.read_text());pid=old['pid'];proc=Path('/proc',str(pid))
    command=(proc/'cmdline').read_bytes().decode().split('\0')
    assert 'poc_flashvep.dense_ar_p_critical_path.attention_poc' in command and 'run' in command
    assert proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid
    assert not (proc/'task'/str(pid)/'children').read_text().strip()
    assert not list((POC/'raw').iterdir()),'Do not stop or replace an already launched PoC'
    lease=json.loads((ROOT/'state/pair_lease_0_1.json').read_text())
    assert not lease.get('name','').startswith('AttentionPoC_')
    target=POC/'state/prelaunch_protocol_v1_revision.json';assert not target.exists()
    write(target,dict(time=time.time(),previous=old,command=command,no_children_verified=True,
        no_GPU_raw_folder=True,pair_owned_by_other_experiment=lease,
        status='STOPPED_PENDING_CONTROLLER_REVISION',
        reason='The exact capture callback rejected the installed backend global-window sentinel(-1,-1). '
            'The immutable V1 config also lacked capture_name required by the benchmark. '
            'Preserve V1 source/config. No GPU job or serving experiment is stopped.'))
    os.kill(pid,signal.SIGTERM)
    return pid


def prepare_v2():
    preserve();assert not (POC/'configs/attention_poc_v2.json').exists()
    assert (POC/'state/prelaunch_protocol_v1_revision.json').exists()
    check=subprocess.run([str(PYTHON),'-m','unittest',
        'poc_flashvep.dense_ar_p_critical_path.test_protocol.ProtocolTests.test_real_capture_seam_accepts_backend_global_window_sentinel'],
        cwd=REPO,capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    parent=json.loads((POC/'configs/attention_poc_v1.json').read_text());archive=POC/'references/attention_poc_source_v2.zip'
    with zipfile.ZipFile(ROOT/parent['source_archive']) as stream:files={name:stream.read(name) for name in stream.namelist()}
    for name in ('attention_capture.py','attention_benchmark.py'):
        files['poc_flashvep/dense_ar_p_critical_path/'+name]=(SOURCE/name).read_bytes()
    for name,value in files.items():
        if name.endswith('.py'):compile(value,name,'exec')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for name,value in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;stream.writestr(item,value)
    cfg=dict(parent,name='Capture_v2',capture_name='Capture_v2',benchmark_prefix='Benchmark_v2',
        source_archive=str(archive.relative_to(ROOT)),source_archive_SHA256=digest(archive),
        protocol_revision='V2 accepts the real global-window sentinel and pins explicit capture/benchmark names. '
            'No prompt,layer,head,iteration,transport,precision or timing contract changes.')
    write(POC/'configs/attention_poc_v2.json',cfg)
    write(POC/'references/source_manifest_v2.json',dict(SHA256=digest(archive),parent_SHA256=parent['source_archive_SHA256'],
        files={name:hashlib.sha256(value).hexdigest() for name,value in files.items()}))
    write(POC/'state/active_protocol.json',dict(version=2,config='attention_poc_v2.json',completion='completion_v2.json'))
    write(POC/'state/prelaunch_API_regression_v2.json',dict(time=time.time(),passed=True,
        command=check.args,stdout=check.stdout,stderr=check.stderr,
        previous_red='The same callback test failed at the V1 None-only sliding-window assertion before the fix.',
        scope='CPU callback seam only. Does not qualify a GPU replay or full-model accuracy.'))
    log=POC/'logs/controller_v2.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.attention_poc','run','2'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(POC/'state/controller_v2.json',dict(time=time.time(),pid=child.pid));preserve();return child.pid


def prepare_v3():
    preserve();assert not (POC/'configs/attention_poc_v3.json').exists()
    parent=json.loads((POC/'configs/attention_poc_v2.json').read_text())
    archive=POC/'references/attention_poc_source_v3.zip'
    with zipfile.ZipFile(ROOT/parent['source_archive']) as stream:files={name:stream.read(name) for name in stream.namelist()}
    files['poc_flashvep/dense_ar_p_critical_path/attention_benchmark.py']=(SOURCE/'attention_benchmark.py').read_bytes()
    for name,value in files.items():
        if name.endswith('.py'):compile(value,name,'exec')
    check=subprocess.run([str(PYTHON),'-m','unittest',
        'poc_flashvep.dense_ar_p_critical_path.test_protocol.ProtocolTests.test_cached_head_transfer_rewrites_only_new_or_partial_KV_pages',
        'poc_flashvep.dense_ar_p_critical_path.test_protocol.ProtocolTests.test_attention_head_split_preserves_complete_GQA_groups'],
        cwd=REPO,capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for name,value in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;stream.writestr(item,value)
    cfg=dict(parent,name='ReuseCapture_v2',reuse_capture=True,parent_completion='completion_v2.json',
        benchmark_prefix='Benchmark_v3',retain_remote_KV_across_chunks=True,
        modes=['LocalFull','LocalSplit','RemoteSplit','RemoteCachedSplit'],remote_modes=['RemoteSplit','RemoteCachedSplit'],
        source_archive=str(archive.relative_to(ROOT)),source_archive_SHA256=digest(archive),
        protocol_revision='Prospective KV-retention intervention. Reuse immutable V2 real-input captures. '
            'RemoteSplit sends the full prefix KV on each chunk. RemoteCachedSplit retains valid per-layer remote KV pages. '
            'Send newly written pages,including a partial boundary page. Preserve every query,key,head and causal position. '
            'Compare both modes with LocalFull within the same three new isolated benchmark processes. '
            'Do not pool earlier V2 processes. No serving/active-decode/full-model accuracy claim.')
    write(POC/'configs/attention_poc_v3.json',cfg)
    write(POC/'references/source_manifest_v3.json',dict(SHA256=digest(archive),parent_SHA256=parent['source_archive_SHA256'],
        files={name:hashlib.sha256(value).hexdigest() for name,value in files.items()}))
    write(POC/'state/KV_retention_protocol_v3.json',dict(time=time.time(),passed_CPU_tests=True,command=check.args,
        stdout=check.stdout,stderr=check.stderr,config_SHA256=digest(POC/'configs/attention_poc_v3.json'),
        primary_isolated_baseline='LocalFull',performance_observed_before_freeze=False))
    log=POC/'logs/controller_v3.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.attention_poc','run','3'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(POC/'state/controller_v3.json',dict(time=time.time(),pid=child.pid));preserve();return child.pid


def reassign_pair67():
    preserve();assert not (POC/'configs/attention_poc_v4.json').exists()
    assert not list((POC/'raw').iterdir()),'Do not move an already launched GPU experiment'
    current=json.loads((ROOT/'state/pair_lease_0_1.json').read_text())
    assert current['status']=='RUNNING' and not current['name'].startswith('AttentionPoC_')
    assert Path('/proc',str(current['pid']),'cmdline').exists(),'Recheck the original live owner before stopping pending CPU controllers'
    target=json.loads((ROOT/'state/pair_lease_6_7.json').read_text());assert target['status']=='RELEASED'
    stopped=[]
    for version in (3,2):
        record=POC/'state'/f'controller_v{version}.json';old=json.loads(record.read_text());pid=old['pid'];proc=Path('/proc',str(pid))
        words=(proc/'cmdline').read_bytes().decode().split('\0')
        assert 'poc_flashvep.dense_ar_p_critical_path.attention_poc' in words and words[-1]==str(version)
        assert proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid
        assert not (proc/'task'/str(pid)/'children').read_text().strip()
        os.kill(pid,signal.SIGTERM)
        state=dict(time=time.time(),status='NOT_RUN',reason='MOVE_UNLAUNCHED_POC_TO_AVAILABLE_PAIR6_7',previous=old,
            verified_no_children=True,verified_no_GPU_raw_folder=True)
        write(POC/'state'/f'completion_v{version}.json',state);stopped.append(state)
    parent=json.loads((POC/'configs/attention_poc_v3.json').read_text())
    cfg=dict(parent,name='Capture_v4',capture_name='Capture_v4',benchmark_prefix='Benchmark_v4',pair=[6,7],reuse_capture=False,
        protocol_revision='Move the still-unlaunched real-input PoC to available H100 pair6/7. '
            'Use the exact V3 capture/benchmark source archive. Preserve every input,head,layer,iteration and KV-retention control. '
            'Leave the original D-cap sweep on GPU0/1 unchanged. Earlier V2/V3 GPU cells remain NOT_RUN,not failed measurements.')
    cfg.pop('parent_completion',None)
    write(POC/'configs/attention_poc_v4.json',cfg)
    write(POC/'state/active_protocol.json',dict(version=4,config='attention_poc_v4.json',completion='completion_v4.json'))
    write(POC/'state/pair_reassignment_v4.json',dict(time=time.time(),source_pair=current,target_pair=target,stopped=stopped,
        worker_source_unchanged=True,reason='The original0/1 controller entered its Discovery lane before the waiting PoC acquired its lease. '
            'Use an available equivalent H100 pair. Stop only childless pending PoC controllers. No serving process is stopped.'))
    log=POC/'logs/controller_v4.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.attention_poc','run','4'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(POC/'state/controller_v4.json',dict(time=time.time(),pid=child.pid));preserve();return child.pid


def prepare_v4_startup_fix():
    import shutil
    preserve();assert not (POC/'configs/attention_poc_v4.json').exists()
    failure=json.loads((POC/'state/completion_v2.json').read_text());assert failure['status']=='FAILED'
    receipt=json.loads((POC/'raw/Capture_v2/dispatcher_result.json').read_text());assert not receipt['complete']
    text=(POC/'logs/Capture_v2.log').read_text();assert "No such file or directory: 'ninja'" in text
    target=json.loads((ROOT/'state/pair_lease_6_7.json').read_text());assert target['status']=='RELEASED'
    for version in (2,3):
        pid=json.loads((POC/'state'/f'controller_v{version}.json').read_text())['pid']
        proc=Path('/proc',str(pid),'cmdline')
        assert not proc.exists() or b'poc_flashvep.dense_ar_p_critical_path.attention_poc' not in proc.read_bytes()
    old=shutil.which('ninja');path=str(PYTHON.parent)+os.pathsep+os.environ['PATH'];ninja=shutil.which('ninja',path=path)
    assert ninja and Path(ninja).is_file()
    parent=json.loads((POC/'configs/attention_poc_v3.json').read_text())
    cfg=dict(parent,name='Capture_v4',capture_name='Capture_v4',benchmark_prefix='Benchmark_v4',pair=[6,7],reuse_capture=False,
        protocol_revision='Repair the proven capture startup PATH failure. Prefix the same environment Python bin used by serving. '
            'Keep the exact V3 worker source and frozen inputs,heads,layers,iterations and KV-retention control. '
            'Use available H100 pair6/7. Preserve the failed V2 capture and the blocked V3 plan. '
            'Limit build parallelism to4. Capture timing is not a performance result.')
    cfg.pop('parent_completion',None)
    write(POC/'configs/attention_poc_v4.json',cfg)
    write(POC/'state/active_protocol.json',dict(version=4,config='attention_poc_v4.json',completion='completion_v4.json'))
    write(POC/'state/startup_PATH_fix_v4.json',dict(time=time.time(),previous_status=failure,previous_receipt=receipt,
        original_ninja_resolvable=old is not None,fixed_ninja=ninja,ninja_SHA256=digest(Path(ninja)),
        worker_source_unchanged=True,target_pair_before_launch=target,
        reason='The isolated controller omitted the Python-bin PATH prefix already used by the standard serving dispatcher. '
            'Do not change the sampler,precision,math,query coverage or failed evidence. No serving process is stopped.'))
    log=POC/'logs/controller_v4.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.attention_poc','run','4'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(POC/'state/controller_v4.json',dict(time=time.time(),pid=child.pid));preserve();return child.pid


if __name__=='__main__':
    import sys
    actions={'stop-pending':stop_pending_v1,'prepare':prepare_v2,'prepare-v3':prepare_v3,
        'pair67':reassign_pair67,'startup-fix-v4':prepare_v4_startup_fix}
    print(actions[sys.argv[1]](),flush=True)
