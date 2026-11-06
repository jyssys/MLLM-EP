"""Check graph and eager coverage before using a collector in serving."""
import csv
import os
from .common import *
from .graph_activity import library,clock


def main():
    configure();cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text());folder=ROOT/'raw'/cfg['name']
    import faulthandler
    faulthandler.enable()
    def mark(stage):
        write(folder/'selftest_progress.json',dict(stage=stage,time=__import__('time').time()))
        print('IDLE_COLLECTOR_SELFTEST',stage,flush=True)
    mark('BEFORE_TORCH')
    import torch
    torch.cuda.set_device(cfg.get('selftest_device',0));value=torch.ones((256,256),device='cuda')
    handle=library();mode=cfg.get('activity_mode',3)
    if mode&2:
        mark('ENABLE_GRAPH_TRACE_BEFORE_GRAPH_CREATION')
        assert handle.activity_begin(mode)==0
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):value.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):value.add_(1)
    torch.cuda.synchronize();before=value[0,0].item()
    mark('GRAPH_READY')
    mark('BEFORE_PRIME_ENABLE')
    if not mode&2:assert handle.activity_begin(mode)==0
    value.add_(1);torch.cuda.synchronize()
    mark('BEFORE_PRIME_PAUSE')
    assert handle.activity_pause()==handle.activity_reset()==0
    mark('PRIMED')
    before=value[0,0].item();assert handle.activity_begin(mode)==0;start=clock(handle)
    count=cfg.get('selftest_launches',100)
    for _ in range(count):value.add_(1)
    for _ in range(count):graph.replay()
    torch.cuda.synchronize();mark('WORK_DRAINED');stop=clock(handle);assert handle.activity_pause()==0
    mark('BEFORE_EXPORT')
    if 'selftest_flush_device' in cfg:torch.cuda.set_device(cfg['selftest_flush_device'])
    assert handle.activity_end(str(folder/'activity.csv').encode())==0
    rows=list(csv.DictReader((folder/'activity.csv').open()))
    kernels=[v for v in rows if int(v['kind'])==10];graphs=[v for v in rows if int(v['kind'])==51]
    result=dict(activity_mode=mode,records=len(rows),graph_envelopes=len(graphs),
        graph_node_kernels=sum(int(v['node'])!=0 for v in kernels),
        non_graph_kernels=sum(int(v['node'])==0 for v in kernels),
        dropped=handle.activity_dropped(),invalid=handle.activity_invalid(),
        output_exact=bool(torch.all(value==before+11*count).item()),start=start,stop=stop,
        collector_counters={key:getattr(handle,'activity_'+key)() for key in
            ('requests','completions','bytes','parsed','version') if hasattr(handle,'activity_'+key)},
        scope='Known graph/eager coverage. Graph envelopes are not exclusive kernel union. No serving-overhead qualification.')
    result['passed']=result['output_exact'] and result['dropped']==result['invalid']==0 and (
        (len(graphs)==count if mode&2 else len(graphs)==0) and (result['non_graph_kernels']==count if mode&1 else True))
    write(folder/'collector_selftest.json',result);print(json.dumps(result));assert result['passed']
    write(folder/'status.json',dict(status='COMPLETE'))


if __name__=='__main__':main()
