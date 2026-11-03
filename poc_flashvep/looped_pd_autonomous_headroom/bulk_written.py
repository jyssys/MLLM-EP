"""Exact interval updates for native host KV-validity bookkeeping.

Graph execution follows vLLM-RLT RecurrentGraphs.execute, source revision
6b164d3844d35ab8e753df1864f0c87dc180401a (Apache-2.0).
See LICENSE.native_graph. The modification groups contiguous host row updates.
GPU kernels, operands, KV writes, clones, and stream events are unchanged.
"""


def add_span(written, start, end):
    """Apply add(position) for every integer in [start, end), exactly."""
    if end <= start or end <= written.prefix:
        return
    if start > written.prefix:
        written.pending.update(range(start, end))
        return
    if written.pending:
        written.pending.difference_update([p for p in written.pending if p < end])
    written.prefix = end
    while written.prefix in written.pending:
        written.pending.remove(written.prefix)
        written.prefix += 1


def mark(cache, rows):
    groups = {}
    for allocation, depth, position in rows:
        key = (id(allocation), depth)
        if key not in groups:
            groups[key] = [allocation, depth, position, position+1]
        else:
            group = groups[key]
            if position != group[3]:
                raise ValueError('Bulk graph bookkeeping requires native contiguous rows')
            group[3] = position+1
    for allocation, depth, start, end in groups.values():
        plane = cache._plane(depth)
        for layer in cache.recurrent_layers:
            add_span(allocation.written[plane][layer], start, end)


def graph_class():
    import torch
    from vllm_rlt.worker.cuda_graph import RecurrentGraphs

    class BulkWrittenGraphs(RecurrentGraphs):
        @torch.inference_mode()
        def execute(self, call, hidden):
            if call.entry is None:
                return super().execute(call, hidden)
            if not call.inputs_ready:
                raise RuntimeError('Graph inputs must be ready before replay')
            call.inputs_ready = False
            batch, entry = call.metadata, call.entry
            count = len(batch.rows)
            entry.graph.replay()
            output = tuple(t[:count].clone() if t is not None else None for t in entry.output)
            self.last_event = torch.cuda.Event()
            self.last_event.record(torch.cuda.current_stream(self.cache.device))
            mark(self.cache, batch.rows)
            self.replays += 1
            return output

    return BulkWrittenGraphs
