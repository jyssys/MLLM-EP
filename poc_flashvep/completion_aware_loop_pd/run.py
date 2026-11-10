"""Launch immutable worker snapshots through the qualified exclusive owner."""
import argparse
import os
import subprocess
import zipfile
from .common import *


def execute(name):
    initialize()
    configuration=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    safety=ROOT/'state/safety_stop_arms.json'
    stops=json.loads(safety.read_text()) if safety.exists() else {}
    arm=configuration.get('primary_safety_arm')
    if arm in stops:
        folder=ROOT/'raw'/name;folder.mkdir(exist_ok=True)
        write(folder/'dispatcher_result.json',dict(status='NOT_RUN_TAIL_SAFETY_STOP',complete=False,
            reason=stops[arm],time=time.time()))
        print('NOT_RUN_TAIL_SAFETY_STOP',name,flush=True)
        return True
    from ..native_split_deep_validation import run as inherited
    inherited.initialize=initialize
    original=subprocess.Popen
    def launch(command,*args,**kwargs):
        target='poc_flashvep.looped_pd_headroom_recovery.worker'
        if isinstance(command,list) and target in command:
            folder=ROOT/'raw'/name;snapshot=folder/'worker_source'
            if configuration.get('full_kernel_trace'):
                if configuration.get('activity_library'):binary=Path(configuration['activity_library'])
                elif configuration.get('full_collector_legacy'):
                    binary=Path(json.loads((ROOT/'tools/activity_build.json').read_text())['library'])
                else:binary=ROOT/'tools/deferred_activity.so'
                source=(REPO/'poc_flashvep/fluidtoken_loop_split/kernel_activity.cc'
                    if configuration.get('full_collector_legacy') else SOURCE/'deferred_activity.cc')
                with zipfile.ZipFile(folder/'full_collector_source_and_binary.zip','w',zipfile.ZIP_DEFLATED) as z:
                    z.write(source,source.name);z.write(binary,binary.name)
                write(folder/'full_collector_manifest.json',dict(binary=str(binary),binary_sha256=digest(binary),
                    source=str(source),source_sha256=digest(source),
                    timed_export='Collector source and binary are frozen before worker startup.'))
            legacy=REPO/'poc_flashvep/idle_intervention_dynamic_loop'
            paths=sorted(legacy.rglob('*.py'))
            with zipfile.ZipFile(folder/'reused_idle_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for path in paths:archive.write(path,str(path.relative_to(REPO)))
            write(folder/'reused_idle_source_hashes.json',{str(path):digest(path) for path in paths})
            with zipfile.ZipFile(folder/'new_campaign_source.zip') as archive:
                assert all(not p.startswith('/') and '..' not in p.split('/') for p in archive.namelist())
                archive.extractall(snapshot)
            env=dict(kwargs['env']);env['INTERVENTION_SNAPSHOT']=str(snapshot)
            env['PYTHONPATH']=str(snapshot/'poc_flashvep/completion_aware_loop_pd/bootstrap')+os.pathsep+env['PYTHONPATH']
            kwargs['env']=env
        return original(command,*args,**kwargs)
    subprocess.Popen=launch
    try:return inherited.execute(name)
    finally:subprocess.Popen=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    raise SystemExit(0 if execute(a.name) else 1)
