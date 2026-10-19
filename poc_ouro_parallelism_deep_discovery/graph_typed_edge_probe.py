"""Read-only typed edges for the same retained graph objects that actually replayed."""
from pathlib import Path
import hashlib
import json

from poc_ouro_parallelism_deep_discovery import graph_debug_probe as debug
from poc_ouro_parallelism_deep_discovery.cuda_graph_typed_edges import ReadOnlyGraphRuntime, inspect_nominal_cuts


class TypedEdgeGraphDebugProbe(debug.GraphDebugProbe):
    def graph_debug_export(self, layers):
        if layers not in (24, 48):
            raise ValueError('Declare the checkpoint physical layer count')
        result = super().graph_debug_export()
        runtime = ReadOnlyGraphRuntime()
        for record in result['records']:
            graph = debug._graphs[record['graph_object_id']]
            raw = int(graph.raw_cuda_graph())
            snapshot = runtime.snapshot(raw)
            dot = Path(record['path'])
            payload = dict(revision=1, graph_object_id=record['graph_object_id'],
                           observed_replays=record['replay_count'], rank=result['rank'],
                           runtime_library=runtime.path, runtime_version=runtime.version,
                           source_dot=str(dot), source_dot_sha256=hashlib.sha256(dot.read_bytes()).hexdigest(),
                           snapshot=snapshot, cut_analysis=inspect_nominal_cuts(snapshot, dot.read_text(), layers),
                           evidence_class='QUALITY_ONLY', timing='NOT_MEASURED',
                           mutation='NONE. No event nodes, dependency changes, or PDL settings.')
            second = runtime.snapshot(raw)
            if int(graph.raw_cuda_graph()) != raw or second['fingerprint'] != snapshot['fingerprint']:
                raise RuntimeError('The graph metadata changed during read-only export')
            payload['metadata_stable_across_two_queries'] = True
            path = dot.with_name(dot.stem + '_edges.json')
            with path.open('x') as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write('\n')
            record['typed_edges'] = dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                nondefault_edge_count=snapshot['nondefault_edge_count'],
                cut_analysis=payload['cut_analysis'], metadata_stable_across_two_queries=True)
        result.update(edge_metadata=True, runtime_library=runtime.path, runtime_version=runtime.version,
                      scope='Read-only typed metadata for actual replay identities. No duration or overlap measurement.')
        return result
