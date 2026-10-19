"""Map immutable typed-edge exports to DOT kernels. This script makes no CUDA call."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from poc_ouro_parallelism_deep_discovery.cuda_graph_typed_edges import (
    CudaGraphEdgeData, decode_edge, inspect_nominal_cuts, snapshot_fingerprint)
from poc_ouro_parallelism_deep_discovery.scripts.parse_native_graph_dot import analyze, parse_graph


def summarize(payload, dot_bytes, layers):
    snapshot = payload['snapshot']
    if hashlib.sha256(dot_bytes).hexdigest() != payload['source_dot_sha256']:
        raise ValueError('The DOT hash differs from the typed export')
    if snapshot_fingerprint(snapshot) != snapshot['fingerprint']:
        raise ValueError('The typed snapshot fingerprint is invalid')
    text = dot_bytes.decode()
    topology, nodes = analyze(text, layers)
    parsed, dot_edges, *_ = parse_graph(text)
    by_handle = {hex(int(node['handle'], 16)): node for node in nodes}
    handles = {identifier: hex(int(node['handle'], 16)) for identifier, node in parsed.items()}
    if set(by_handle) != {node['handle'] for node in snapshot['nodes']}:
        raise ValueError('The DOT and typed node identities differ')
    if Counter((handles[a], handles[b]) for a, b in dot_edges) != Counter(
            (edge['source'], edge['target']) for edge in snapshot['edges']):
        raise ValueError('The DOT and typed endpoint multiplicities differ')
    if sorted(edge['api_index'] for edge in snapshot['edges']) != list(range(len(snapshot['edges']))):
        raise ValueError('Typed edge indices are missing or duplicated')
    mapped = []
    for edge in sorted(snapshot['edges'], key=lambda value: value['api_index']):
        raw = bytes.fromhex(edge['raw_hex'])
        if len(raw) != 8:
            raise ValueError('An edge record does not contain eight bytes')
        decoded = decode_edge(CudaGraphEdgeData.from_buffer_copy(raw))
        if any(edge[key] != value for key, value in decoded.items()):
            raise ValueError('Named edge fields disagree with the raw bytes')
        source, target = by_handle[edge['source']], by_handle[edge['target']]
        row = dict(api_index=edge['api_index'], source_handle=edge['source'], target_handle=edge['target'],
                   source_id=source['id'], target_id=target['id'],
                   source_position=source['position'], target_position=target['position'],
                   source_kernel=source['kernel'], target_kernel=target['kernel'],
                   source_classification=source['classification'], target_classification=target['classification'],
                   **decoded)
        mapped.append(row)
    nondefault = [edge for edge in mapped if not edge['is_default']]
    cut_analysis = inspect_nominal_cuts(snapshot, text, layers)
    pair_counts = Counter((edge['source_classification'], edge['target_classification']) for edge in nondefault)
    summary = dict(node_count=len(nodes), edge_count=len(mapped),
        node_kinds=topology['node_kinds'], endpoint_topology_single_chain=topology['single_chain'],
        metadata_counts=dict(Counter(edge['raw_hex'] for edge in mapped)),
        default_edge_count=len(mapped) - len(nondefault), nondefault_edge_count=len(nondefault),
        reserved_nonzero_count=sum(edge['reserved_nonzero'] for edge in mapped),
        nondefault_classification_pairs=[dict(source=a, target=b, count=count)
                                        for (a, b), count in sorted(pair_counts.items())],
        cut_analysis=cut_analysis, nondefault_edges=nondefault,
        snapshot_fingerprint=snapshot['fingerprint'], observed_replays=payload['observed_replays'],
        timing='NOT_MEASURED', overlap_duration_ms=None, kernel_overlap='NOT_MEASURED',
        interpretation='Endpoint connectivity is not a full-kernel serialization claim. Typed ports and dependencies remain intact.')
    return summary, mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', type=Path, required=True)
    parser.add_argument('--dot', type=Path)
    parser.add_argument('--layers', type=int, choices=(24, 48), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload_bytes = args.edges.read_bytes()
    payload = json.loads(payload_bytes)
    dot = args.dot or Path(payload['source_dot'])
    summary, rows = summarize(payload, dot.read_bytes(), args.layers)
    summary.update(source_edges=str(args.edges.resolve()), source_dot=str(dot.resolve()),
                   source_edges_sha256=hashlib.sha256(payload_bytes).hexdigest(),
                   source_dot_sha256=payload['source_dot_sha256'], evidence_class='CPU_ANALYSIS_OF_CAPTURED_METADATA')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write('\n')
    with (args.output / 'edges_with_kernels.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('nondefault_edges', 'cut_analysis')}, sort_keys=True))


if __name__ == '__main__':
    main()
