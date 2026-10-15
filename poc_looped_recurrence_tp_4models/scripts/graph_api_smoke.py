"""Validate retained graph introspection without loading model weights."""
import os
from types import SimpleNamespace
import torch
from poc_looped_recurrence_tp_4models.graph_probe import GraphProbe
from poc_looped_recurrence_tp_4models.contract import write


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    x=torch.ones(16,device='cuda');stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):y=x+1;z=y*2
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):y=x+1;z=y*2
    graph.replay();torch.cuda.synchronize();assert bool((z==4).all())
    probe=GraphProbe();probe.model_runner=SimpleNamespace(cudagraph_manager=SimpleNamespace(graphs={'smoke':graph}))
    records=probe.graph_nodes();assert len(records)==1 and records[0]['node_count']>0
    from pathlib import Path
    write(Path(os.environ['LOOPED_KV_OUTPUT'])/'graph_api.json',dict(success=True,records=records))
    print(records,flush=True)


if __name__=='__main__':main()
