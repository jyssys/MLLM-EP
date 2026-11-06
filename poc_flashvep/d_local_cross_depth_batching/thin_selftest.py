"""Validate primed, paused collection. This is not a model performance run."""
import csv
import os
from .common import *
from .thin_activity import library, prime, clock


def main():
    configure();cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text());folder=ROOT/'raw'/cfg['name']
    import torch
    torch.cuda.set_device(0)
    value=torch.ones((256,256),device='cuda')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):value.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):value.add_(1)
    torch.cuda.synchronize();before=value[0,0].item()
    handle=library();prime(handle,torch.device('cuda:0'))
    assert handle.activity_begin()==0;start=clock(handle)
    left,right=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    left.record()
    for _ in range(100):value.add_(1)
    for _ in range(100):graph.replay()
    right.record();right.synchronize();stop=clock(handle)
    assert handle.activity_pause()==0
    # Unrecorded work can continue before the deferred export.
    value.add_(1);torch.cuda.synchronize()
    assert handle.activity_end(str(folder/'kernel_activity.csv').encode())==0
    rows=list(csv.DictReader((folder/'kernel_activity.csv').open()))
    graph_count=sum(int(r['graph_node'])!=0 for r in rows)
    result=dict(expected=200,recorded=len(rows),graph_recorded=graph_count,
        dropped=handle.activity_dropped(),invalid=handle.activity_invalid(),
        output_exact=bool(torch.all(value==before+201).item()),start=start,stop=stop,
        passed=len(rows)==200 and graph_count==100 and handle.activity_dropped()==handle.activity_invalid()==0,
        scope='Collector self-test with eager/graph work, priming, pause and deferred flush. No model or serving measurement.')
    write(folder/'thin_selftest.json',result);assert result['passed'] and result['output_exact']
    write(folder/'status.json',dict(status='COMPLETE'))


if __name__=='__main__':main()
