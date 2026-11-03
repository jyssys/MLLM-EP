"""Warm one real D-local prompt before measured arrivals in both controls."""
import time
import json
from .common import ROOT, write


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt import SamplingParams
    from ..looped_pd_atomic_prefill import rlt_worker
    previous_stats = StageEngine.stats
    previous_step = StageEngine.step
    previous_serve = rlt_worker.serve
    folder = ROOT/'raw'/cfg['name']
    request = folder/'D_probe_warmup_requested.json'
    finished = folder/'D_probe_warmup_finished.json'
    rid = 'autonomous_probe_geometry_warmup'
    state = dict(started=False, begin=None)
    count = cfg['D_probe_warmup_tokens']
    tokens = json.loads((ROOT/'data'/cfg['dataset']).read_text())[0]['input_ids'][cfg['model']][:count]
    assert len(tokens) == count

    def stats(engine):
        if engine.stage_config.role == 'decode' and request.exists() and not state['started']:
            state.update(started=True, begin=time.monotonic())
            engine._add_request(rid, tokens, SamplingParams(max_tokens=1, temperature=0.,
                seed=cfg['seed']+800000, min_loops=cfg['depth'], max_loops=cfg['depth'],
                exit_threshold=1., ignore_eos=True))
        return previous_stats(engine)

    def step(engine):
        normal = []
        for output in previous_step(engine):
            if output.request_id != rid:
                normal.append(output)
            elif output.finished:
                assert len(output.token_ids) == 1 and list(output.exit_depths) == [cfg['depth']]
                write(finished, dict(start=state['begin'], finish=time.monotonic(),
                    input_ids=tokens, output_ids=list(output.token_ids), exit_depths=list(output.exit_depths),
                    scope='Real full-depth warm-up only. Excluded from measured work, arrivals, throughput, and SLO.'))
        return normal

    def serve(engine, inputs, args, run, warmup=False, contract=None):
        result = previous_serve(engine, inputs, args, run, warmup, contract)
        if warmup:
            write(request, dict(time=time.monotonic(), tokens=count))
            deadline = time.monotonic()+120.
            while not finished.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('D-local probe warm-up did not complete')
                for peer in engine.peers.values():
                    peer.client.stats()
                assert not engine.step(), 'Unconsumed output after native warm-up'
                time.sleep(.01)
        return result

    StageEngine.stats = stats
    StageEngine.step = step
    rlt_worker.serve = serve
