"""Supplement declared DAG identities. This is not a model or runtime proof."""
from dataclasses import dataclass

from poc_ouro_parallelism_deep_discovery.scripts.verify_dag import validate


@dataclass(frozen=True)
class KVState:
    request_id: str
    generation: int
    r: int
    l: int
    first_position: int
    end_exclusive: int
    version: str = 'committed'
    committed: bool = True


def generation(node):
    value = node.metadata.get('generation', 0)
    assert type(value) is int and value >= 0, 'Invalid declared generation'
    return value


def identity(node):
    return node.request_id, generation(node), node.token_or_chunk_id


def validate_identities(nodes, *, phase='decode', external_kv=None, known_tokens=None):
    """Check explicit witnesses. Do not infer missing full-model dependencies.

    Known-token origins are caller-declared prompt IDs or prior committed samples.
    External history must be registered with its exact identity and range.
    This function does not observe actual values, sampling, original-gate arithmetic,
    KV allocation/eviction, or hardware resource capacity. It rejects cross-depth
    aliases and provisional token speculation unless another validator models them.
    """
    assert phase in ('prefill', 'decode'), 'Declare prefill or decode explicitly'
    external_kv = {} if external_kv is None else external_kv
    known_tokens = {} if known_tokens is None else known_tokens
    for node in nodes:
        assert isinstance(node.request_id, str) and node.request_id, 'Invalid node request identity'
        assert all(type(index) is int and index >= 0 for index in
                   [node.token_or_chunk_id, node.r, node.l]), 'Invalid node identity indices'
        assert type(node.committed) is bool, 'Invalid committed-state declaration'
    for key, origin in known_tokens.items():
        assert (isinstance(key, tuple) and len(key) == 3 and isinstance(key[0], str)
                and type(key[1]) is int and key[1] >= 0
                and type(key[2]) is int and key[2] >= 0), 'Invalid known-token identity'
        assert origin in ('PROMPT', 'PRIOR_COMMITTED_SAMPLE'), 'Known-token origin is not authoritative'
    for state in external_kv.values():
        assert isinstance(state, KVState), 'Register typed KV state'
        assert isinstance(state.request_id, str) and state.request_id, 'Invalid KV request identity'
        assert all(type(index) is int for index in [state.generation, state.r, state.l,
                   state.first_position, state.end_exclusive]), 'KV identity indices must be integers'
        assert type(state.committed) is bool, 'Invalid KV committed-state declaration'
        assert (min(state.generation, state.r, state.l, state.first_position) >= 0
                and state.end_exclusive > state.first_position), 'Invalid KV range or identity'
    validate(nodes)
    lookup = {node.node_id: node for node in nodes}
    ancestors = {}
    while len(ancestors) < len(nodes):
        ready = [node for node in nodes if node.node_id not in ancestors
                 and set(node.predecessors) <= ancestors.keys()]
        assert ready, 'Unresolved ancestor graph'
        for node in ready:
            ancestors[node.node_id] = set(node.predecessors)
            for predecessor in node.predecessors:
                ancestors[node.node_id].update(ancestors[predecessor])
    produced = {key: node for node in nodes for key in node.kv_writes}
    assert not (set(external_kv) & set(produced)), 'External KV cannot also have an in-graph writer'
    for node in nodes:
        for key in node.kv_reads:
            if key.startswith('history:'):
                assert key in external_kv, 'Unregistered external history'
                state = external_kv[key]
                assert state.committed and state.version == 'committed', 'History is not a committed version'
                assert state.end_exclusive <= node.token_or_chunk_id, 'Historical KV includes a future/current position'
            else:
                producer = produced[key]
                assert producer.node_id in ancestors[node.node_id], 'KV producer needs an explicit dependency path'
                assert not node.committed or producer.committed, 'Committed consumer reads provisional KV'
                assert (producer.committed or producer.scratch_version == node.scratch_version), 'Provisional KV branch mismatch'
                state = KVState(producer.request_id, generation(producer), producer.r, producer.l,
                    producer.token_or_chunk_id, producer.token_or_chunk_id + 1,
                    version=producer.scratch_version, committed=producer.committed)
                assert state.end_exclusive <= node.token_or_chunk_id + 1, 'KV includes a future position'
            assert (state.request_id, state.generation, state.r, state.l) == (
                node.request_id, generation(node), node.r, node.l), 'KV identity mismatch'
        if node.metadata.get('requires_online_gate'):
            expected_depth = node.r - 1
            candidates = [lookup[key] for key in ancestors[node.node_id] if lookup[key].operator == 'gate']
            assert any(identity(gate) == identity(node) and gate.r == expected_depth
                       and gate.committed and gate.scratch_version == 'committed'
                       for gate in candidates), 'Matching gate identity/depth is unavailable'
    for node in nodes:
        if identity(node) in known_tokens:
            continue
        if phase == 'prefill':
            raise AssertionError('Token input is not registered as a known prompt chunk')
        witnesses = [lookup[key] for key in ancestors[node.node_id] if lookup[key].operator == 'sample']
        assert any(sample.request_id == node.request_id and generation(sample) == generation(node)
                   and sample.token_or_chunk_id + 1 == node.token_or_chunk_id
                   and sample.committed and sample.scratch_version == 'committed'
                   for sample in witnesses), 'Token input lacks a matching committed sample predecessor'
    return True
