"""Synthetic graph tracing self-test. Not model cost or serving evidence."""
import argparse,json,time
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('baseline','early','late','serial','graph'));p.add_argument('--namespace',default='collector_smoke');args=p.parse_args()
    folder=ROOT/'raw'/(args.namespace+'_'+args.mode);assert not folder.exists();folder.mkdir(parents=True)
    from .activity import library,stamp
    import torch
    h=None
    if args.mode in ('early','serial','graph'):
        h,info=library();assert h.activity_kernel_mode(int(args.mode=='serial'))==0
        assert h.activity_graph_mode(int(args.mode=='graph'))==0
        assert h.activity_begin(1024**3,1)==0
    torch.cuda.set_device(0);torch.manual_seed(2026111200)
    x=torch.randn((64,1024),device='cuda',dtype=torch.bfloat16)
    w=torch.randn((1024,1024),device='cuda',dtype=torch.bfloat16)/32
    def execute():
        y=x
        for _ in range(32):y=torch.relu(torch.mm(y,w))
        return y
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):execute()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=execute()
    if args.mode=='late':h,info=library();assert h.activity_begin(1024**3,1)==0
    begin_stamp=stamp(h) if h else None
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();begin=time.monotonic()
    for _ in range(1000):graph.replay()
    if args.mode=='graph':out.add_(1)
    b.record();b.synchronize();wall=time.monotonic()-begin;ms=a.elapsed_time(b)
    end_stamp=stamp(h) if h else None
    capture=None
    if h:
        code=h.activity_end(str(folder/'kernels.csv').encode(),begin_stamp['cupti_ns'],end_stamp['cupti_ns'])
        capture=dict(status=code,records=h.activity_count(),dropped=h.activity_dropped(),invalid=h.activity_invalid(),
            all_kernels=h.activity_all_kernels(),requested_buffers=h.activity_requested_buffers(),start=begin_stamp,stop=end_stamp)
        if args.mode=='graph':
            import csv
            rows=list(csv.DictReader((folder/'kernels.csv').open()))
            counts={kind:sum(r['kind']==kind for r in rows) for kind in ('graph','kernel')}
            capture.update(kind_counts=counts,all_graphs=h.activity_all_graphs())
            assert counts['graph']==1000 and counts['kernel']>=1,capture
        else:assert capture['records']>=64000,capture
        assert code==0 and capture['dropped']==capture['invalid']==0,capture
    value=dict(mode=args.mode,complete=True,synthetic_graph_replays=1000,expected_graph_kernels=64000,
        wall_seconds=wall,CUDA_Event_enclosure_ms=ms,capture=capture,finite_output=bool(torch.isfinite(out).all()),
        scope='Collector functional self-test with synthetic BF16 operands. Not real recurrent weights, Attention, model cost, online overhead, or idle capacity.')
    write(folder/'result.json',value);print(json.dumps(value))


if __name__=='__main__':main()
