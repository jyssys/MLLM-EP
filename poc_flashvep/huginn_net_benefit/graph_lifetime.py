"""Destroy drained graph executables before the Native NIXL agent closes."""
import os
from .common import ROOT, write


def reset_entries(cache, seen):
    count = 0
    if cache is None:
        return count
    for entry in cache.entries.values():
        graph = entry.graph
        if graph is not None and id(graph) not in seen:
            seen.add(id(graph))
            graph.reset()
            count += 1
        entry.graph = None
    return count


def install(cfg):
    from ..native_loop_2d_oracle import native_path
    connector = native_path.ACTIVE_CONNECTOR
    previous = connector.close

    def closed(owner):
        assert not owner.has_pending, 'Graph cleanup requires no retained Native transfer'
        runner = owner.engine.model_runner
        runner.synchronize()
        seen, reset = set(), {}
        prompt = getattr(owner.engine, '_autonomous_prefill_graphs', None)
        green = getattr(runner, 'net_prompt_green', None)
        if green is not None:
            green.set_context()
        try:
            reset['prompt'] = reset_entries(prompt, seen)
        finally:
            if green is not None:
                green.pop_context()
        decode = getattr(runner, 'net_decode_green', None)
        if decode is not None:
            decode.set_context()
        try:
            reset['decode'] = reset_entries(runner.graphs, seen)
            reset['coda'] = reset_entries(runner.coda_graphs, seen)
        finally:
            if decode is not None:
                decode.pop_context()
        write(ROOT/'raw'/cfg['name']/f'graph_lifetime_cleanup_{os.getpid()}.json', dict(
            role=owner.engine.stage_config.role, reset_executables=reset,
            no_pending_transfers=True,
            scope='Post-serving cleanup after Native GPU drain and before NIXL deregistration. This does not alter measured model execution, buffers, or Native ACK ordering. Graph audit snapshots precede this reset.'))
        return previous(owner)
    connector.close = closed
