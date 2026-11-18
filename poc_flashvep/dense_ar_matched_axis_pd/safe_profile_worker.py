"""Start collection before measurement ACK and stop after worker completion."""
import atexit
import json
import os
from pathlib import Path
import sys
import time
from .common import ROOT,write


def install(cfg,role):
    if not cfg.get('nsys_bounded_capture'):
        return
    import torch
    from ..dense_ar_slo_layer_pd.native import BatchChannel
    folder=ROOT/'raw'/cfg['name']
    state=dict(start=None,end=None,start_API_host_ms=None,stop_API_host_ms=None)
    ancestry=[];pid=os.getpid()
    while pid>1 and len(ancestry)<16:
        try:
            text=Path(f'/proc/{pid}/status').read_text()
            fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in text.splitlines() if ':' in line}
            parent=int(fields['PPid'])
            ancestry.append(dict(pid=pid,parent=parent,session=os.getsid(pid),
                uid=int(fields['Uid'].split()[0]),executable=os.path.basename(os.readlink(f'/proc/{pid}/exe'))))
            pid=parent
        except (FileNotFoundError,ProcessLookupError):
            break
    write(folder/f'profiler_owned_process_{role}.json',dict(pid=os.getpid(),session=os.getsid(0),
        uid=os.getuid(),ancestry=ancestry,config=os.environ['DENSE_CONFIG'],
        source_root=os.environ.get('DENSE_FROZEN_ROOT'),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scope='Record process ancestry before measurement. Do not record arbitrary environment variables or ancestor command arguments. This does not relax the original ownership guard.'))
    def marker(kind):
        before=time.monotonic()
        torch.cuda.nvtx.range_push(f'ELASTIC_MONO_{kind}_{before:.9f}')
        after=time.monotonic();torch.cuda.nvtx.range_pop()
        return dict(before=before,after=after)
    def start():
        if state['start'] is not None:return
        before=time.monotonic()
        torch.cuda.cudart().cudaProfilerStart()
        state['start_API_host_ms']=1000*(time.monotonic()-before)
        state['start']=marker('START')
    def stop():
        if state['start'] is None or state['end'] is not None:return
        state['end']=marker('END');before=time.monotonic()
        torch.cuda.cudart().cudaProfilerStop()
        state['stop_API_host_ms']=1000*(time.monotonic()-before)
        write(folder/f'nsys_clock_{role}.json',dict(state,
            scope='cudaProfilerStart finishes before the measurement-ready ACK. cudaProfilerStop runs when the worker closes after all request outputs. No benchmark input, model math or partition is changed. API durations are host enclosures, not CUDA kernel time.'))
    receive=BatchChannel.recv
    close=BatchChannel.close
    def received(self,*args,**kwargs):
        value=receive(self,*args,**kwargs)
        if value['kind']=='measurement_begin':start()
        return value
    def closed(self,*args,**kwargs):
        stop()
        return close(self,*args,**kwargs)
    BatchChannel.recv=received
    BatchChannel.close=closed
    atexit.register(stop)


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text())
    from ..dense_ar_slo_layer_pd import nsys_window,native
    nsys_window.install=install
    native.main()


if __name__=='__main__':
    main()
