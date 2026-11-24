"""Bind audited runtime helpers in this process without editing old files."""
import importlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from .common import ROOT,PRIOR,V2,REPO,BASE,BRANCH,START,PYTHON,initialize,preserve,resources,write,digest


def time_available(field='deadline_epoch',budget=None):
    return time.time()<(budget or json.loads((ROOT/'state/budget.json').read_text()))[field]


def dispatch_allowed(budget=None):
    return not (ROOT/'state/STOP_DISPATCH.json').exists() and time_available('last_large_dispatch_epoch',budget)


def execution_allowed(budget=None):
    return not (ROOT/'state/STOP_EXECUTION.json').exists() and time_available('deadline_epoch',budget)


def copy_data(name):
    target=ROOT/'data'/name
    parent=next((parent for parent in (PRIOR,V2) if (parent/'data'/name).exists()),None)
    assert parent is not None,name
    source=parent/'data'/name
    if not target.exists():shutil.copyfile(source,target)
    assert digest(target)==digest(source),name
    return target


def bind():
    initialize()
    pc=importlib.import_module('poc_flashvep.dense_ar_p_critical_path.common')
    values=dict(ROOT=ROOT,PRIOR=V2,BASE=BASE,BRANCH=BRANCH,START=START,initialize=initialize,
        preserve=preserve,resources=resources,copy_data=copy_data,time_available=time_available,
        dispatch_allowed=dispatch_allowed,execution_allowed=execution_allowed)
    for key,value in values.items():setattr(pc,key,value)
    runner=pc.bind()
    for name,module in list(sys.modules.items()):
        if name.startswith('poc_flashvep.dense_ar_p_critical_path.'):
            for key,value in values.items():
                if hasattr(module,key):setattr(module,key,value)
    return runner


def source_capsule(name):
    initialize();source=PRIOR/'references/source_capsules'/name
    target=ROOT/'references/source_capsules'/('ns_'+name)
    if target.exists():return target
    with zipfile.ZipFile(source) as archive:
        files={key:archive.read(key) for key in archive.namelist()}
    key='poc_flashvep/dense_ar_longbench400_pd/common.py'
    before="ROOT=REPO/'poc_flashvep/reports/dense_ar_p_critical_path_20261123'"
    after="ROOT=REPO/'poc_flashvep/reports/dense_ar_native_ttft_joint_slo_20261124'"
    assert files[key].decode().count(before)==1
    files[key]=files[key].decode().replace(before,after).encode()
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for filename,data in sorted(files.items()):
            entry=zipfile.ZipInfo(filename,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            bundle.writestr(entry,data)
    write(target.with_suffix('.json'),dict(time=time.time(),parent=str(source.relative_to(REPO)),
        parent_SHA256=digest(source),SHA256=digest(target),changed=[key+':ROOT_LITERAL_ONLY'],
        files=len(files),all_other_source_bytes_identical=True,
        scope='Only the output namespace changes. Keep the inherited numerical,compiled Stock,KV,ACK,credit and scheduler implementations. '
            'New runtime policy combinations still require fresh correctness and quality gates.'))
    return target


def start_guardian():
    initialize();path=ROOT/'state/guardian_controller.json'
    if path.exists():return json.loads(path.read_text())
    code='from poc_flashvep.dense_ar_native_ttft_slo.runtime import bind; bind(); from poc_flashvep.dense_ar_output_aware_v2.bridge import main; main()'
    command=[str(PYTHON),'-u','-c',code,'guardian']
    with (ROOT/'logs/guardian.log').open('w') as stream:
        child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(time=time.time(),pid=child.pid,command=command)
    write(path,value);return value


if __name__=='__main__':
    if sys.argv[1]=='guardian':print(start_guardian(),flush=True)
