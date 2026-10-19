"""Identity witnesses supplement the retained partial structural checker."""
from dataclasses import replace
import unittest

from poc_ouro_parallelism_deep_discovery.dag_identity_checks import KVState, validate_identities
from poc_ouro_parallelism_deep_discovery.scripts.verify_dag import Node, graph


def node(name, request='q0', token=0, r=0, l=0, operator='embed', predecessors=None, **kwargs):
    return Node(name, request, token, r, l, operator, predecessors or [], None,
                kwargs.pop('reads', []), kwargs.pop('writes', []), 0, **kwargs)


class IdentityTests(unittest.TestCase):
    def test_unregistered_history_rejected(self):
        consumer = node('a', token=1, operator='attention', reads=['history:other'])
        with self.assertRaisesRegex(AssertionError, 'Unregistered'):
            validate_identities([consumer], known_tokens={('q0', 0, 1): 'PRIOR_COMMITTED_SAMPLE'})

    def test_registered_matching_history_positive(self):
        consumer = node('a', token=1, operator='attention', reads=['history:registered'])
        state = KVState('q0', 0, 0, 0, 0, 1)
        self.assertTrue(validate_identities([consumer], external_kv={'history:registered': state},
            known_tokens={('q0', 0, 1): 'PRIOR_COMMITTED_SAMPLE'}))

    def test_history_identity_and_future_position_rejected(self):
        consumer = node('a', token=1, operator='attention', reads=['history:registered'])
        original = KVState('q0', 0, 0, 0, 0, 1)
        for field, value in [('request_id', 'other'), ('generation', 1), ('r', 1), ('l', 1),
                             ('end_exclusive', 2), ('committed', False)]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_identities([consumer], external_kv={'history:registered': replace(original, **{field: value})},
                    known_tokens={('q0', 0, 1): 'PRIOR_COMMITTED_SAMPLE'})

    def test_future_decode_input_needs_sample(self):
        with self.assertRaisesRegex(AssertionError, 'Token input'):
            validate_identities([node('future', token=1)], known_tokens={('q0', 0, 0): 'PROMPT'})

    def test_sample_predecessor_positive(self):
        sample = node('sample0', operator='sample')
        future = node('future', token=1, predecessors=['sample0'])
        self.assertTrue(validate_identities([sample, future], known_tokens={('q0', 0, 0): 'PROMPT'}))

    def test_other_request_or_generation_sample_rejected(self):
        for request, generation in [('other', 0), ('q0', 1)]:
            sample = node('s', request=request, operator='sample', metadata={'generation': generation})
            future = node('future', token=1, predecessors=['s'])
            with self.subTest(request=request, generation=generation), self.assertRaisesRegex(AssertionError, 'Token input'):
                validate_identities([sample, future], known_tokens={
                    (request, generation, 0): 'PROMPT', ('q0', 0, 0): 'PROMPT'})

    def test_prediction_is_not_known_input(self):
        with self.assertRaisesRegex(AssertionError, 'Known-token origin'):
            validate_identities([node('future', token=1)], known_tokens={('q0', 0, 1): 'PREDICTED'})

    def test_gate_requires_matching_request_depth_and_generation(self):
        for changes in [{'request_id': 'other'}, {'r': 0}, {'token_or_chunk_id': 1},
                        {'metadata': {'generation': 1}}]:
            gate = replace(node('gate', r=1, operator='gate'), **changes)
            consumer = node('continue', r=2, operator='next_pass', predecessors=['gate'],
                            metadata={'requires_online_gate': True})
            known = {('q0', 0, 0): 'PROMPT', (gate.request_id, gate.metadata.get('generation', 0),
                                            gate.token_or_chunk_id): 'PROMPT'}
            with self.subTest(changes=changes), self.assertRaisesRegex(AssertionError, 'Matching gate'):
                validate_identities([gate, consumer], known_tokens=known)

    def test_matching_gate_positive(self):
        gate = node('gate', r=1, operator='gate')
        consumer = node('continue', r=2, operator='next_pass', predecessors=['gate'],
                        metadata={'requires_online_gate': True})
        self.assertTrue(validate_identities([gate, consumer], known_tokens={('q0', 0, 0): 'PROMPT'}))

    def test_cross_request_current_KV_rejected(self):
        producer = node('producer', request='other', operator='norm_qkv', writes=['kv'])
        consumer = node('consumer', operator='attention', predecessors=['producer'], reads=['kv'])
        with self.assertRaisesRegex(AssertionError, 'KV identity'):
            validate_identities([producer, consumer], known_tokens={('q0', 0, 0): 'PROMPT', ('other', 0, 0): 'PROMPT'})

    def test_current_KV_needs_dependency_not_incidental_topological_order(self):
        producer = node('producer', operator='norm_qkv', writes=['kv'])
        independent = node('independent')
        delay = node('delay', predecessors=['independent'])
        consumer = node('consumer', operator='attention', predecessors=['delay'], reads=['kv'])
        with self.assertRaisesRegex(AssertionError, 'explicit dependency'):
            validate_identities([producer, independent, delay, consumer], known_tokens={('q0', 0, 0): 'PROMPT'})

    def test_committed_consumer_cannot_read_provisional_current_KV(self):
        producer = node('producer', operator='norm_qkv', writes=['kv'],
                        committed=False, scratch_version='branch1')
        consumer = node('consumer', operator='attention', predecessors=['producer'], reads=['kv'])
        with self.assertRaisesRegex(AssertionError, 'provisional KV'):
            validate_identities([producer, consumer], known_tokens={('q0', 0, 0): 'PROMPT'})

    def test_same_private_branch_positive(self):
        producer = node('producer', operator='norm_qkv', writes=['kv'],
                        committed=False, scratch_version='branch1')
        consumer = node('consumer', operator='attention', predecessors=['producer'], reads=['kv'],
                        committed=False, scratch_version='branch1')
        self.assertTrue(validate_identities([producer, consumer], known_tokens={('q0', 0, 0): 'PROMPT'}))

    def test_different_private_branch_rejected(self):
        producer = node('producer', operator='norm_qkv', writes=['kv'],
                        committed=False, scratch_version='branch1')
        consumer = node('consumer', operator='attention', predecessors=['producer'], reads=['kv'],
                        committed=False, scratch_version='branch2')
        with self.assertRaisesRegex(AssertionError, 'branch mismatch'):
            validate_identities([producer, consumer], known_tokens={('q0', 0, 0): 'PROMPT'})

    def test_declared_prefill_graph_positive(self):
        nodes = graph(layers=2, loops=4, chunks=3, P=2)
        self.assertTrue(validate_identities(nodes, phase='prefill',
            known_tokens={('q0', 0, index): 'PROMPT' for index in range(3)}))

    def test_external_history_cannot_also_have_an_in_graph_writer(self):
        producer = node('p', operator='norm_qkv', writes=['history:shared'],
                        committed=False, scratch_version='branch1')
        consumer = node('c', token=1, operator='attention', reads=['history:shared'])
        with self.assertRaisesRegex(AssertionError, 'External KV.*writer'):
            validate_identities([producer, consumer],
                external_kv={'history:shared': KVState('q0', 0, 0, 0, 0, 1)},
                known_tokens={('q0', 0, 0): 'PROMPT', ('q0', 0, 1): 'PRIOR_COMMITTED_SAMPLE'})

    def test_history_identity_fields_require_integral_indices(self):
        consumer = node('a', token=1, operator='attention', reads=['history:registered'])
        original = KVState('q0', 0, 0, 0, 0, 1)
        for field in ['generation', 'r', 'l', 'first_position', 'end_exclusive']:
            for value in [0.5, 0.0, True]:
                with self.subTest(field=field, value=value), self.assertRaises(AssertionError):
                    validate_identities([consumer],
                        external_kv={'history:registered': replace(original, **{field: value})},
                        known_tokens={('q0', 0, 1): 'PRIOR_COMMITTED_SAMPLE'})


if __name__ == '__main__':
    unittest.main()
