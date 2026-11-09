"""Warm exact shapes with real startup requests. Never count these as adaptation."""
import time
from .common import ROOT,write


def install(cfg):
    from ..looped_pd_atomic_prefill import rlt_worker
    from ..looped_pd_headroom_recovery.graph_warmup import worker_stats
    from vllm_rlt import SamplingParams
    previous=rlt_worker.serve
    folder=ROOT/'raw'/cfg['name']
    def served(engine,inputs,args,run,warmup=False,contract=None):
        result=previous(engine,inputs,args,run,warmup,contract)
        if not warmup:return result
        assert not (folder/'measurement_begin.json').exists()
        # Drain the ordinary startup requests before shape-only work.
        begin=time.monotonic()
        while engine.transfers:
            assert time.monotonic()-begin<180,'Startup ACK drain timed out'
            assert not engine.step()
        rows=[]
        geometries=[[width]*batch for width in cfg['intervention_warmup_widths']
            for batch in cfg.get('intervention_warmup_batches',[1])]
        if cfg.get('intervention_warmup_mixed_native'):
            assert cfg.get('native_frontier') and cfg['native_frontier'][0]==0
            geometries.extend([width,0] for width in cfg['intervention_warmup_widths'])
        for index,widths in enumerate(geometries):
            width,batch=widths[0],len(widths)
            before=worker_stats(engine);started=time.monotonic()
            ids={f'idle_shape_warm_{index}_{member}_{w}' for member,w in enumerate(widths)}
            for member,rid in enumerate(sorted(ids)):
                engine.add_request(rid,list(inputs[0]),SamplingParams(max_tokens=1,temperature=0.,
                    seed=cfg['seed']-50000-index*8-member,min_loops=cfg['depth'],max_loops=cfg['depth'],
                    exit_threshold=1.,ignore_eos=True))
            finished=set()
            while finished!=ids:
                assert time.monotonic()-started<180,'Exact shape startup timed out'
                for output in engine.step():
                    assert output.request_id in ids
                    if output.finished:
                        assert len(output.token_ids)==1 and set(output.exit_depths)=={cfg['depth']}
                        finished.add(output.request_id)
            while engine.transfers:
                assert time.monotonic()-started<180,'Exact shape ACK drain timed out'
                assert not engine.step()
            after=worker_stats(engine)
            work=sum(value['issued_loop_rows']-before[key]['issued_loop_rows'] for key,value in after.items())
            assert work==batch*cfg['length']*cfg['depth'],'Startup work was missing or duplicated'
            rows.append(dict(width=width,batch=batch,widths_per_request=widths,wall_seconds=time.monotonic()-started,
                full_recurrent_rows=work,passed=True))
            write(folder/'deterministic_shape_warmup.json',dict(rows=rows,
                scope='Real exact startup requests. Excluded from measured requests, offload counts, latency, and throughput.'))
        return result
    rlt_worker.serve=served
