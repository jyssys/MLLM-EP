"""Add optional observation and one explicit safe-retirement control."""
import hashlib
import json
import shutil
import zipfile
from .common import ROOT,PRIOR,SOURCE,write,digest,preserve
from .bootstrap import reroot


def replace_once(value,before,after):
    assert value.count(before)==1,('Frozen source seam changed',before)
    return value.replace(before,after)


def build(original_source,revision='r1'):
    preserve();source=reroot(original_source)
    target=source.with_name(source.stem+'_trace_retirement_'+revision+'.zip')
    if target.exists():return target
    with zipfile.ZipFile(source) as archive:files={name:archive.read(name) for name in archive.namelist()}
    worker='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py';value=files[worker].decode()
    value=replace_once(value,'    adapter=None\n',
        '    from poc_flashvep.dense_ar_p_critical_path.instrumentation import install as install_pc_trace\n'
        '    pc_trace=install_pc_trace(cfg,role,folder,info)\n    pc_retirement=None\n    adapter=None\n')
    value=replace_once(value,'    def ack(self,value):\n        transfers.append',
        '    def ack(self,value):\n        pc_before=pc_retirement.before_ack(self,value) if pc_retirement is not None else {}\n        transfers.append')
    value=replace_once(value,'        return original_ack(self,value)\n',
        '        result=original_ack(self,value)\n'
        '        if pc_retirement is not None:pc_retirement.after_ack(self,pc_before)\n        return result\n')
    value=replace_once(value,'        pool=scheduler.kv_cache_manager.block_pool;initial_free=pool.get_num_free_blocks()\n',
        '        from poc_flashvep.dense_ar_p_critical_path.retirement import Retirement\n'
        '        pc_retirement=Retirement(cfg,role,folder,adapter,channel)\n'
        '        pool=scheduler.kv_cache_manager.block_pool;initial_free=pool.get_num_free_blocks()\n')
    value=replace_once(value,"                    channel.send(dict(kind='boundary_ready',role=role));continue\n",
        "                    if pc_trace is not None:pc_trace.boundary()\n"
        "                    channel.send(dict(kind='boundary_ready',role=role));continue\n")
    value=replace_once(value,'            combined_waiting=list(scheduler.waiting)+list(scheduler.skipped_waiting)\n',
        '            if pc_retirement is not None:pc_retirement.tick(scheduler)\n'
        '            combined_waiting=list(scheduler.waiting)+list(scheduler.skipped_waiting)\n')
    value=replace_once(value,'        if adapter is not None:adapter.close()\n',
        '        if pc_trace is not None:pc_trace.close()\n'
        '        if pc_retirement is not None:pc_retirement.close()\n'
        '        if adapter is not None:adapter.close()\n')
    files[worker]=value.encode()
    coordinator='poc_flashvep/dense_ar_longbench400_pd/serving.py';value=files[coordinator].decode()
    value=replace_once(value,"'P_wait_segments.jsonl','request_submissions.jsonl'):",
        "'P_wait_segments.jsonl','request_submissions.jsonl','source_retirement_notifications.jsonl'):")
    needle="                    if message['kind']=='accepted':\n"
    inserted=("                    if message['kind']=='source_resources_retired':\n"
        "                        assert role=='prefill' and cfg.get('source_credit_release_on_retirement')\n"
        "                        assert all(message[k] for k in ('native_request_removed','hidden_ACKed','hidden_owner_released','native_fence_complete'))\n"
        "                        before=len(source_owned)\n"
        "                        from poc_flashvep.dense_ar_p_critical_path.retirement import release_owned\n"
        "                        released=release_owned(source_owned,rid)\n"
        "                        append(folder/'source_retirement_notifications.jsonl',dict(message,received=time.monotonic(),credit_released=released))\n"
        "                        if released:append(folder/'credit_events.jsonl',dict(event='RELEASE_SAFE_SOURCE_RETIREMENT',request_id=rid,time=time.monotonic(),owned_before=before))\n"
        "                        continue\n")
    value=replace_once(value,needle,inserted+needle);files[coordinator]=value.encode()
    for name in ('__init__.py','instrumentation.py','retirement.py'):
        files['poc_flashvep/dense_ar_p_critical_path/'+name]=(SOURCE/name).read_bytes()
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    write(target.with_suffix('.json'),dict(SHA256=digest(target),parent_SHA256=digest(source),
        changed=[worker,coordinator],added=['P model-call observer','native-retirement/hidden-ACK coordinator credit control'],
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Optional trace flags only observe original call boundaries. The retirement switch is a separate runtime intervention. '
            'No manual KV free,output correction,attention change or D active-decode policy change. '
            'A new source/signature needs its explicit correctness proof before primary performance.'))
    return target


def collector():
    source=PRIOR/'references/kernel_only_collector_v3';target=ROOT/'references/kernel_only_collector_v3'
    target.mkdir(parents=True,exist_ok=True);record=json.loads((source/'build.json').read_text())
    for key in ('source','library'):
        path=__import__('pathlib').Path(record[key]);destination=target/path.name
        if not destination.exists():shutil.copyfile(path,destination)
        assert digest(destination)==record[key+'_SHA256'];record[key]=str(destination)
    write(target/'build.json',record)
