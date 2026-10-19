"""Check dependency and ownership claims without pretending synthetic costs are timings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Node:
    node_id: str
    request_id: str
    token_or_chunk_id: int
    r: int
    l: int
    operator: str
    predecessors: list[str]
    weight_id: str | None
    kv_reads: list[str]
    kv_writes: list[str]
    owner: int
    layout: str = 'full_hidden'
    scratch_version: str = 'committed'
    estimated_cost: float = 1.0
    resource_type: str = 'compute'
    committed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

def validate(nodes: list[Node]):
    lookup = {n.node_id:n for n in nodes}
    assert len(lookup) == len(nodes), 'Duplicate node identity'
    produced = {}
    for n in nodes:
        for key in n.kv_writes:
            assert key not in produced, 'A KV version has multiple writers'
            produced[key] = n.node_id
        if n.committed:
            assert n.scratch_version == 'committed', 'A provisional branch cannot publish committed state'
    done = set()
    while len(done) < len(nodes):
        ready = [n for n in nodes if n.node_id not in done and set(n.predecessors) <= done]
        assert ready, 'Cycle or unavailable predecessor'
        for n in ready:
            for key in n.kv_reads:
                assert key.startswith('history:') or (key in produced and produced[key] in done), 'Unavailable KV version'
            if n.weight_id is not None:
                assert n.metadata.get('weight_resident', False), 'Required weight is not resident'
            if n.metadata.get('requires_online_gate'):
                assert any(lookup[p].operator == 'gate' for p in n.predecessors), 'Future exit decision used without a computed gate'
        done.update(n.node_id for n in ready)
    return True

def graph(layers=24, loops=4, chunks=4, P=4, placement='physical', requests=1):
    nodes=[]
    for req in range(requests):
        request=f'q{req}'
        for r in range(loops):
            for l in range(layers):
                for j in range(chunks):
                    prefix=f'{request}:r{r}:l{l}:c{j}'
                    if l:
                        deps=[f'{request}:r{r}:l{l-1}:c{j}:mlp']
                    elif r:
                        deps=[f'{request}:r{r-1}:boundary:c{j}']
                    else:
                        deps=[]
                    owner=(l*P//layers if placement=='physical' else r*P//loops)%P
                    key=f'{request}:r{r}:l{l}:c{j}:kv'
                    resident={'weight_resident':True, 'weights_alias_across_r':True}
                    nodes.append(Node(prefix+':qkv',request,j,r,l,'norm_qkv',deps,
                        f'W:l{l}:qkv',[],[key],owner,metadata=resident))
                    prior=[f'{request}:r{r}:l{l}:c{k}:qkv' for k in range(j+1)]
                    reads=[f'{request}:r{r}:l{l}:c{k}:kv' for k in range(j+1)]
                    nodes.append(Node(prefix+':attention',request,j,r,l,'attention',prior,
                        f'W:l{l}:attention',reads,[],owner,metadata=resident))
                    nodes.append(Node(prefix+':mlp',request,j,r,l,'norm_mlp_residual',[prefix+':attention'],
                        f'W:l{l}:mlp',[],[],owner,metadata=resident))
            for j in range(chunks):
                nodes.append(Node(f'{request}:r{r}:boundary:c{j}',request,j,r,layers,'loop_norm',
                    [f'{request}:r{r}:l{layers-1}:c{j}:mlp'],'W:loop_norm',[],[],
                    (layers-1)*P//layers if placement=='physical' else r*P//loops,
                    metadata={'weight_resident':True}))
    return nodes

def schedule(nodes, transfer_cost=.0):
    """One greedy feasible resource schedule. It is not an optimum or hardware result."""
    validate(nodes)
    lookup={n.node_id:n for n in nodes}
    finish={}; lanes={}; records=[]
    remaining=set(lookup)
    while remaining:
        ready=[lookup[key] for key in remaining if set(lookup[key].predecessors)<=finish.keys()]
        options=[]
        for n in ready:
            arrival=max([finish[p]+(transfer_cost if lookup[p].owner!=n.owner else 0.) for p in n.predecessors] or [0.])
            start=max(arrival,lanes.get((n.owner,n.resource_type),0.))
            options.append((start,n.node_id,n))
        start,_,n=min(options,key=lambda v:(v[0],v[1]))
        end=start+n.estimated_cost
        lanes[n.owner,n.resource_type]=end
        finish[n.node_id]=end
        records.append({'node_id':n.node_id,'owner':n.owner,'start_arbitrary_units':start,'end_arbitrary_units':end})
        remaining.remove(n.node_id)
    return {'duration_arbitrary_units':max(finish.values()),'schedule':records,
        'evidence_class':'SCHEDULE_HEURISTIC','cost_scope':'Synthetic unit costs. Not milliseconds or GPU speedup.',
        'transfer_scope':'Fixed per-edge synthetic delay. Link contention is not calibrated in this structural test.'}

def unrolled_signature(nodes,layers):
    """Map (r,l) to the logical depth while retaining every norm and KV version."""
    names={n.node_id:(n.request_id,n.token_or_chunk_id,n.r*layers+n.l,n.operator) for n in nodes}
    return sorted((names[n.node_id],tuple(sorted(names[p] for p in n.predecessors)),
        tuple(n.kv_reads),tuple(n.kv_writes),n.weight_id) for n in nodes)

class Tests(unittest.TestCase):
    def test_legal_prefill(self):
        self.assertTrue(validate(graph(layers=2,loops=4,chunks=3,P=2)))
    def test_missing_dependency_rejected(self):
        nodes=graph(layers=2,loops=2,chunks=2,P=2)
        nodes[0].predecessors=['not_ready']
        with self.assertRaisesRegex(AssertionError,'unavailable predecessor'): validate(nodes)
    def test_wrong_KV_rejected(self):
        nodes=graph(layers=2,loops=2,chunks=2,P=2)
        nodes[1].kv_reads=['different_request:r3:l9:c0:kv']
        with self.assertRaisesRegex(AssertionError,'Unavailable KV'): validate(nodes)
    def test_unresident_weight_rejected(self):
        nodes=graph(layers=2,loops=1,chunks=1,P=2)
        nodes[0].metadata['weight_resident']=False
        with self.assertRaisesRegex(AssertionError,'not resident'): validate(nodes)
    def test_provisional_commit_rejected(self):
        nodes=graph(layers=2,loops=1,chunks=1,P=2)
        nodes[0].scratch_version='losing_branch'
        with self.assertRaisesRegex(AssertionError,'provisional branch'): validate(nodes)
    def test_online_gate_rejected(self):
        nodes=graph(layers=2,loops=1,chunks=1,P=2)
        nodes[0].metadata['requires_online_gate']=True
        with self.assertRaisesRegex(AssertionError,'Future exit'): validate(nodes)
    def test_decode_chain_one_ready_task(self):
        nodes=graph(layers=2,loops=4,chunks=1,P=4)
        result=schedule(nodes)
        self.assertEqual(result['duration_arbitrary_units'],len(nodes))
    def test_unroll_retains_aliases(self):
        nodes=graph(layers=2,loops=4,chunks=3,P=2)
        signature=unrolled_signature(nodes,2)
        self.assertEqual(len(signature),len(nodes))
        self.assertEqual(len({n.weight_id for n in nodes}),7)

def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful(): raise SystemExit(1)
    target=ROOT/'artifacts/dag_checks'
    target.mkdir(parents=True,exist_ok=True)
    matrix=[]
    for layers in [24,48]:
        for P in [2,4,8]:
            for chunks in [1,2,4,8]:
                for placement in ['physical','depth']:
                    nodes=graph(layers=layers,loops=4,chunks=chunks,P=P,placement=placement)
                    result=schedule(nodes)
                    row={'layers':layers,'R':4,'P':P,'chunks':chunks,'placement':placement,
                        'node_count':len(nodes),'duration_arbitrary_units':result['duration_arbitrary_units'],
                        'physical_GPU_execution':False,'evidence_class':'SCHEDULE_HEURISTIC',
                        'aliased_unroll_signature_sha256':hashlib.sha256(repr(unrolled_signature(nodes,layers)).encode()).hexdigest()}
                    matrix.append(row)
                    if layers==24 and P==4 and chunks==4 and placement=='physical':
                        (target/'example_nodes.json').write_text(json.dumps([asdict(n) for n in nodes],indent=2)+'\n')
                        (target/'example_schedule.json').write_text(json.dumps(result,indent=2)+'\n')
    (target/'structural_matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    print('DAG_CHECKS_PASS. Costs are synthetic, not measured GPU latency.')

if __name__=='__main__':main()
