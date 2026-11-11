import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from .monotone_early_protocol import derive, early_receipt_admissible as admissible, qualification_receipt


class MonotoneReceiptTests(unittest.TestCase):
    def setUp(self):
        self.owner = NS(cfg={'depth':32}, net_immutable_receipts={
            'A':dict(depth=16, KV_only=True, full_prompt=2048)})
        self.transfer = NS(tid='A', received={0,1}, outgoing=False, cancelled=False,
            ready=False, expected=None, cut=32, request=NS(prompt_token_ids=[0]*2048,
                phase=NS(value='receiving')))

    def test_empty_receive_set_retains_original_policy_semantics(self):
        self.transfer.received = set()
        self.assertTrue(admissible(self.owner, self.transfer, 28))

    def test_only_monotone_complete_prompt_KV_is_admissible(self):
        self.assertTrue(admissible(self.owner, self.transfer, 28))
        self.assertFalse(admissible(self.owner, self.transfer, 15))
        self.assertFalse(admissible(self.owner, self.transfer, 32))
        self.assertFalse(admissible(self.owner, self.transfer, True))

    def test_received_data_without_a_frontier_is_not_admissible(self):
        self.owner.net_immutable_receipts.clear()
        self.assertFalse(admissible(self.owner, self.transfer, 28))

    def test_hidden_or_partial_prompt_proof_is_not_admissible(self):
        for field, value in (('KV_only', False), ('full_prompt', 1024)):
            original = self.owner.net_immutable_receipts['A'][field]
            self.owner.net_immutable_receipts['A'][field] = value
            self.assertFalse(admissible(self.owner, self.transfer, 28))
            self.owner.net_immutable_receipts['A'][field] = original

    def test_committed_activated_cancelled_or_outgoing_state_is_not_admissible(self):
        for field, value in (('expected', 10), ('cut', 28), ('ready', True),
                ('cancelled', True), ('outgoing', True)):
            original = getattr(self.transfer, field)
            setattr(self.transfer, field, value)
            self.assertFalse(admissible(self.owner, self.transfer, 28))
            setattr(self.transfer, field, original)
        self.transfer.request.phase.value = 'prefilling_recurrent'
        self.assertFalse(admissible(self.owner, self.transfer, 28))

    def test_derivation_changes_only_the_two_named_policy_predicates(self):
        source = (Path(__file__).parents[1]/'completion_aware_loop_pd/runtime.py').read_text()
        derived, lines = derive(source)
        self.assertEqual(len(lines), 2)
        tree = ast.parse(derived)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'early_receipt_admissible']
        self.assertEqual(len(calls), 2)
        self.assertIn("'STALE_OR_EXPIRED_COMMIT'", derived)
        self.assertIn('completion.feasible', derived)
        self.assertIn('self.completion_quantum_floor', derived)

    def test_measured_commit_ignores_contract_files_and_warmup(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            def save(name, value):
                (folder/name).write_text(json.dumps(value))
            save('measurement_begin.json', dict(time=10.))
            save('intervention_contract_1.json', dict(role='decode'))
            record = dict(event='EARLY_RECEIPT_POLICY_CHECK', kind='idle_commit',
                          immutable_frontier_valid=True, received_fragments=1)
            save('immutable_early_receipts_1.json', dict(records=[
                dict(record, time=5., tid='warm'), dict(record, time=15., tid='live')]))
            save('intervention_1.json', dict(records=[
                dict(event='DESTINATION_COMMIT', time=5.1, tid='warm'),
                dict(event='DESTINATION_COMMIT', time=15.1, tid='live')]))
            result = qualification_receipt(folder)
            self.assertTrue(result['passed'])
            self.assertEqual(result['committed_after_immutable_receipt'], 1)


if __name__ == '__main__':
    unittest.main()
