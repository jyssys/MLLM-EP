"""Process-local graph topology capture. No model arithmetic replacement."""
from pathlib import Path
import os
import torch
from poc_ouro_parallelism_deep_discovery.worker_probe import CampaignProbe

_graphs={}
_replays={}
_observe=False
_original_new=torch.cuda.CUDAGraph.__new__
_original_init=torch.cuda.CUDAGraph.__init__
_original_replay=torch.cuda.CUDAGraph.replay


def _new(cls,keep_graph=False):
    # Torch2.11 capture_end clears has_graph_ when keep_graph is false,
    # even when debug mode prevented destruction of the raw graph.
    # This diagnostic retains the raw graph explicitly. It is not timing.
    return _original_new(cls,keep_graph=True)


def _init(graph,*args,**kwargs):
    assert len(args)<=1 and set(kwargs)<= {'keep_graph'}
    _original_init(graph,True)
    graph.enable_debug_mode()
    _graphs[id(graph)]=graph


def _replay(graph,*args,**kwargs):
    result=_original_replay(graph,*args,**kwargs)
    if _observe:_replays[id(graph)]=_replays.get(id(graph),0)+1
    return result


torch.cuda.CUDAGraph.__new__=staticmethod(_new)
torch.cuda.CUDAGraph.__init__=_init
torch.cuda.CUDAGraph.replay=_replay


class GraphDebugProbe(CampaignProbe):
    def graph_debug_begin(self):
        global _observe
        torch.cuda.synchronize()
        _replays.clear();_observe=True
        return dict(created_graph_objects=len(_graphs),scope='Observe actual replay identities. No model-layer hook.')

    def graph_debug_export(self):
        global _observe
        from vllm.distributed import get_tensor_model_parallel_rank
        _observe=False;torch.cuda.synchronize()
        rank=get_tensor_model_parallel_rank();output=Path(os.environ['LOOPED_KV_OUTPUT'])
        records=[]
        for ordinal,(identifier,count) in enumerate(sorted(_replays.items())):
            path=output/f'rank{rank}_replayed_graph{ordinal}.dot'
            assert not path.exists()
            _graphs[identifier].debug_dump(str(path))
            assert path.exists(),str(path)
            records.append(dict(graph_object_id=identifier,replay_count=count,path=str(path),bytes=path.stat().st_size))
        return dict(rank=rank,records=records,total_observed_replays=sum(_replays.values()),
                    scope='Actual replayed graph topology only. Debug mode retains graph state. No node timing or E2E attribution.')
