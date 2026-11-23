"""Change coordinator evidence I/O only. Preserve every worker byte."""
import hashlib
import zipfile
from .common import ROOT,SOURCE,write,digest
from .capsule import replace_once


def build(parent_name,revision='r1'):
    assert revision in ('r1','r2')
    parent=ROOT/'references/source_capsules'/parent_name;target=parent.with_name(parent.stem+'_coordinator_io_'+revision+'.zip')
    if target.exists():return target
    with zipfile.ZipFile(parent) as archive:files={name:archive.read(name) for name in archive.namelist()}
    path='poc_flashvep/dense_ar_longbench400_pd/serving.py';original=files[path];value=original.decode()
    before="""    if path.name in ('credit_events.jsonl','P_wait_segments.jsonl','request_submissions.jsonl','source_retirement_notifications.jsonl'):
        BUFFERED.setdefault(path,[]).append(row)
        return
    with path.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\\n')
"""
    value=replace_once(value,'BUFFERED={}\n','BUFFERED={}\nPC_COORDINATOR=None\n')
    value=replace_once(value,before,'    return PC_COORDINATOR.append(path,row,BUFFERED)\n')
    value=replace_once(value,"    cfg_path=Path(sys.argv[1]);cfg=json.loads(cfg_path.read_text());folder=ROOT/'raw'/cfg['name']\n",
        "    cfg_path=Path(sys.argv[1]);cfg=json.loads(cfg_path.read_text());folder=ROOT/'raw'/cfg['name']\n"
        "    global PC_COORDINATOR\n"
        "    from poc_flashvep.dense_ar_p_critical_path.coordinator_probe import install\n"
        "    PC_COORDINATOR=install(cfg,folder)\n")
    value=replace_once(value,'        sender=Ingress(channels)\n',
        '        channels={role:PC_COORDINATOR.channel(channel,role) for role,channel in channels.items()}\n'
        '        sender=Ingress(channels)\n')
    value=replace_once(value,'        while i<len(data) or len(completed)<len(data):\n            now=time.monotonic()\n',
        '        while i<len(data) or len(completed)<len(data):\n            now=time.monotonic()\n            PC_COORDINATOR.loop(now)\n')
    value=replace_once(value,'        BUFFERED.clear()\n','        BUFFERED.clear()\n        PC_COORDINATOR.close()\n')
    if revision=='r2':
        value=replace_once(value,"        append(folder/'measurement_begin.jsonl',", 
            "        PC_COORDINATOR.begin_measurement()\n        append(folder/'measurement_begin.jsonl',")
        value=replace_once(value,"        write(folder/'status.json',dict(status='COMPLETE',time=time.time(),completed=len(completed)))\n",
            "        PC_COORDINATOR.end_measurement()\n"
            "        write(folder/'status.json',dict(status='COMPLETE',time=time.time(),completed=len(completed)))\n")
    files[path]=value.encode();files['poc_flashvep/dense_ar_p_critical_path/coordinator_probe.py']=(SOURCE/'coordinator_probe.py').read_bytes()
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(item,data)
    write(target.with_suffix('.json'),dict(SHA256=digest(target),parent_SHA256=digest(parent),
        changed=[path],added=['coordinator_probe.py'],all_worker_bytes_identical=True,
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        scope='Only coordinator evidence append/flush and optional host observation change. '
            'Revision2 also provides an optional coordinator-only cyclic-GC deferral. It never changes worker GC. '
            'Existing asynchronous sender threads remain unchanged. Runtime hidden/KV ACK files are not buffered. '
            'No model,sampling,client-clock,arrival,output-cap,worker-admission or GPU policy change. '
            'Apply the intervention to Native,Layer,Stock Token and Hybrid. It is not a layer-specific kernel improvement.'))
    return target
