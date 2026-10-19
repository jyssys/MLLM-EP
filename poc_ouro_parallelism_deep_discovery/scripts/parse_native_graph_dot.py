"""Inspect a captured CUDA graph. Topology is not a latency measurement."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


NODE = re.compile(r'"(graph_\d+_node_(\d+))"\[.*?label="\{\s*(\w+)\n(.*?)\n\}"];', re.S)
EDGE = re.compile(r'"(graph_\d+_node_\d+)" -> "(graph_\d+_node_\d+)"')
DETAIL = re.compile(r'\| \{ID \| (\d+) \(topoId: (\d+)\) \| (.*?)\}.*?node handle \| func handle\} \| \{(0x[0-9A-Fa-f]+) \| (0x[0-9A-Fa-f]+)\}', re.S)


def parse_graph(text):
    nodes = {}
    for match in NODE.finditer(text):
        identifier, ordinal, kind, body = match.groups()
        detail = DETAIL.search(body)
        if not detail and kind != 'MEMCPY':
            raise ValueError(f'Unrecognized node detail: {identifier}')
        if detail:
            local_id, topo_id, kernel, handle, function = detail.groups()
        else:
            copy_id = re.search(r'\{(\d+) \(topoId: (\d+)\) \| (0x[0-9A-Fa-f]+)\}', body)
            if not copy_id:
                raise ValueError(f'Unrecognized memcpy node: {identifier}')
            local_id, topo_id, handle = copy_id.groups()
            kernel, function = 'MEMCPY', ''
        if identifier in nodes or int(ordinal) != int(local_id):
            raise ValueError('Duplicate or inconsistent node ID')
        nodes[identifier] = dict(id=identifier, ordinal=int(ordinal), kind=kind,
                                 topo_id=int(topo_id), kernel=kernel,
                                 handle=handle, function_handle=function)
    edges = list(EDGE.findall(text))
    if not nodes or len(edges) != len(set(edges)):
        raise ValueError('Empty graph or duplicate edge')
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for source, target in edges:
        if source not in nodes or target not in nodes:
            raise ValueError('An edge references an unknown node')
        incoming[target].append(source)
        outgoing[source].append(target)
    indegree = {key: len(incoming[key]) for key in nodes}
    frontier = sorted(key for key, degree in indegree.items() if not degree)
    order, frontier_sizes = [], []
    while frontier:
        frontier_sizes.append(len(frontier))
        key = frontier.pop(0)
        order.append(key)
        for target in outgoing[key]:
            indegree[target] -= 1
            if not indegree[target]:
                frontier.append(target)
        frontier.sort()
    if len(order) != len(nodes):
        raise ValueError('The graph contains a cycle')
    return nodes, edges, order, incoming, outgoing, frontier_sizes


def classify(kernel):
    if kernel == 'MEMCPY':
        return 'MEMCPY_UNRESOLVED_PURPOSE'
    if 'reshape_and_cache' in kernel:
        return 'CURRENT_KV_WRITE'
    if 'FlashAttnFwdCombine' in kernel:
        return 'ATTENTION_SPLIT_MERGE'
    if 'FlashAttnFwdSm90' in kernel:
        return 'ATTENTION_CORE'
    if 'splitKreduce' in kernel:
        return 'GEMM_SPLITK_REDUCTION'
    if 'nvjet_' in kernel:
        return 'GEMM_UNRESOLVED_ROLE'
    if 'rms_norm' in kernel:
        return 'FUSED_NORM_RESIDUAL'
    if 'silu' in kernel:
        return 'MLP_ACTIVATION'
    if 'cat_index_select' in kernel:
        return 'QKV_ROPE_LAYOUT'
    return 'UNRESOLVED'


def analyze(text, layers):
    nodes, edges, order, incoming, outgoing, widths = parse_graph(text)
    sequence = [nodes[key] for key in order]
    chain = (len(edges) == len(nodes) - 1
             and all(len(incoming[key]) <= 1 and len(outgoing[key]) <= 1 for key in nodes))
    rows = []
    for position, node in enumerate(sequence):
        rows.append(dict(position=position, **node, classification=classify(node['kernel'])))
    cache_positions = [i for i, row in enumerate(rows) if row['classification'] == 'CURRENT_KV_WRITE']
    summary = dict(node_count=len(nodes), edge_count=len(edges),
                   node_kinds=dict(Counter(node['kind'] for node in sequence)),
                   classification_counts=dict(Counter(row['classification'] for row in rows)),
                   roots=[key for key in nodes if not incoming[key]],
                   tails=[key for key in nodes if not outgoing[key]],
                   single_chain=chain, max_ready_nodes_in_captured_dag=max(widths),
                   cache_write_nodes=len(cache_positions), physical_layers=layers,
                   expected_R4_cache_writes=4 * layers,
                   timing='NOT_MEASURED. Node counts and edges do not determine kernel duration.',
                   dependency_scope='Captured execution order only. A stream edge can be stronger than a mathematical data dependency.',
                   recurrence_scope='Repeated kernels do not imply reusable activation values or independent recurrent passes.')
    if chain and len(cache_positions) == 4 * layers:
        cuts = [cache_positions[r * layers] - 3 for r in range(4)]
        # The first operation of a layer is a fused normalization kernel.
        # Verify this observed pattern. Do not infer a cut from layer counts alone.
        valid_cuts = (cuts[0] == 0 and all(cut >= 0 and rows[cut]['classification'] == 'FUSED_NORM_RESIDUAL' for cut in cuts))
        summary['verified_repeated_pattern'] = valid_cuts
        if valid_cuts:
            summary['logical_pass_kernel_cuts'] = cuts + [len(rows)]
            summary['cut_scope'] = ('Cuts precede the observed first-layer normalization kernels. '
                                    'A boundary kernel can fuse preceding-pass normalization and next-pass input work. '
                                    'These cuts are not exact Python recurrent-pass completion timestamps.')
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dot', type=Path, required=True)
    parser.add_argument('--layers', type=int, choices=[24, 48], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.dot.read_bytes()
    summary, rows = analyze(payload.decode(), args.layers)
    args.output.mkdir(parents=True, exist_ok=False)
    summary.update(source=str(args.dot.resolve()), source_sha256=hashlib.sha256(payload).hexdigest())
    (args.output/'topology_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    with (args.output/'nodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
