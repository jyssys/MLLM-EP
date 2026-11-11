"""Reuse the qualified dispatcher and freeze both new and reused operator code."""
import argparse,zipfile,threading,time,signal,json
from .common import *


def execute(name):
    initialize();assert not (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    from ..huginn_net_benefit import run as reused
    reused.ROOT=ROOT;reused.initialize=initialize
    original=subprocess.Popen
    def launch(command,*args,**kwargs):
        if isinstance(command,list) and 'poc_flashvep.looped_pd_headroom_recovery.worker' in command:
            folder=ROOT/'raw'/name;snapshot=folder/'worker_source'
            paths=sorted(p for source in (SOURCE,REPO/'poc_flashvep/loop_width_gemm_validation') for p in source.rglob('*') if p.suffix in ('.py','.cc'))
            with zipfile.ZipFile(folder/'discovery_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for path in paths:archive.write(path,str(path.relative_to(REPO)))
                archive.extractall(snapshot)
            write(folder/'discovery_source_hashes.json',{str(p.relative_to(REPO)):digest(snapshot/p.relative_to(REPO)) for p in paths})
            env=dict(kwargs['env']);env['PREFILL_DISCOVERY_SNAPSHOT']=str(snapshot)
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if cfg.get('discovery_activity'):
                # Compile the immutable worker copy before the GPU process starts.
                # A later edit cannot substitute another collector binary.
                from .activity import build
                record=build(snapshot/'poc_flashvep/prefill_heavy_decode_safe/preallocated_activity.cc')
                assert digest(record['library'])==record['library_SHA256']
                assert digest(snapshot/'poc_flashvep/prefill_heavy_decode_safe/preallocated_activity.cc')==record['source_SHA256']
                write(folder/'activity_runtime_build.json',record)
            if cfg.get('discovery_cupti_preload'):
                library=Path(cfg['discovery_cupti_preload'])
                assert digest(library)==cfg['discovery_cupti_preload_SHA256']
                assert not env.get('LD_PRELOAD'),'Do not replace an unrelated preload'
                env['LD_PRELOAD']=str(library)
            env['PYTHONPATH']=str(snapshot/'poc_flashvep/prefill_heavy_decode_safe/bootstrap')+os.pathsep+env['PYTHONPATH']
            kwargs['env']=env
        process=original(command,*args,**kwargs)
        if isinstance(command,list) and 'poc_flashvep.looped_pd_headroom_recovery.worker' in command:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if cfg.get('discovery_activity'):
                def watch():
                    folder=ROOT/'raw'/name;signature=None;progress=time.monotonic()
                    from ..looped_pd_headroom_recovery.run import measured_requests_complete
                    while process.poll() is None:
                        path=folder/'requests.jsonl';current=path.stat().st_size if path.exists() else None
                        if current!=signature:signature=current;progress=time.monotonic()
                        if signature and not measured_requests_complete(folder,cfg['count']) and time.monotonic()-progress>120:
                            assert os.getsid(process.pid)==process.pid
                            assert Path('/proc',str(process.pid)).stat().st_uid==os.getuid()
                            for stack in folder.glob('worker_stacks_*.log'):
                                pid=int(stack.stem.rsplit('_',1)[1])
                                try:
                                    if os.getsid(pid)==process.pid:os.kill(pid,signal.SIGUSR1)
                                except ProcessLookupError:pass
                            write(folder/'discovery_progress_watchdog.json',dict(time=time.time(),
                                reason='PROFILE_NO_OUTPUT_PROGRESS_120S',PID=process.pid,request_file_bytes=signature,
                                scope='Only the verified owned session is stopped. Failed captures cannot supply physical-capacity evidence.'))
                            time.sleep(.5);os.killpg(process.pid,signal.SIGTERM);return
                        time.sleep(2)
                threading.Thread(target=watch,daemon=True).start()
        return process
    subprocess.Popen=launch
    try:return reused.execute(name)
    finally:subprocess.Popen=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');raise SystemExit(0 if execute(p.parse_args().name) else 1)
