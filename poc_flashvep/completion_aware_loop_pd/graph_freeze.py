"""Use cached graphs or eager execution. Never capture a cold measured graph."""
import os
from types import MethodType
from .common import ROOT,write


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.cuda_graph import PreparedGraph
    previous_init=StageEngine.__init__;previous_close=StageEngine.close;previous_stats=StageEngine.stats
    folder=ROOT/'raw'/cfg['name'];marker=folder/'measurement_begin.json'

    def initialize(engine,*args,**kwargs):
        result=previous_init(engine,*args,**kwargs)
        graphs=getattr(engine,'_autonomous_prefill_graphs',None)
        if graphs is None:return result
        previous=graphs.select;state=dict(cold_measured_eager_fallbacks=0,
            measured_cached_graph_selections=0,measured_original_eager_selections=0)
        def select(owner,hidden,metadata):
            call=previous(hidden,metadata)
            if marker.exists() and call.entry is not None and call.entry.graph is None:
                state['cold_measured_eager_fallbacks']+=1
                return PreparedGraph(metadata)
            if marker.exists():
                key='measured_cached_graph_selections' if call.entry is not None else 'measured_original_eager_selections'
                state[key]+=1
            return call
        graphs.select=MethodType(select,graphs)
        engine._completion_graph_freeze=state
        return result

    def close(engine):
        if hasattr(engine,'_completion_graph_freeze'):
            audit=getattr(engine,'_deep_graph_audit',None)
            captures=audit['counts']['measured']['captures'] if audit is not None else None
            write(folder/f'graph_freeze_{os.getpid()}.json',dict(role=engine.stage_config.role,
                **engine._completion_graph_freeze,measured_packed_graph_captures=captures,
                scope='Packed-prefill cache selections, not all model graph calls. All arms use the same cache budget and startup policy. Cold measured shapes use the qualified eager path.'))
            assert captures in (0,None),'A cold packed graph was captured during measurement'
        return previous_close(engine)

    def stats(engine,*args,**kwargs):
        if cfg.get('minimum_HBM_free_bytes') and not marker.exists():
            import torch
            free,total=torch.cuda.mem_get_info(engine.model_runner.device)
            if free<cfg['minimum_HBM_free_bytes']:
                write(folder/f'HBM_guard_failure_{os.getpid()}.json',dict(free_bytes=free,total_bytes=total,
                    required_free_bytes=cfg['minimum_HBM_free_bytes'],role=engine.stage_config.role))
                raise RuntimeError('Startup HBM reserve is below the declared guard')
        return previous_stats(engine,*args,**kwargs)

    StageEngine.__init__=initialize;StageEngine.close=close;StageEngine.stats=stats
