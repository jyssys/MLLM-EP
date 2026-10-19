"""Characterize the existing checker. Do not replace it or validate Ouro models."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "contract_scope_existing_dag_v1", ROOT / "scripts/verify_dag.py")
DAG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DAG
SPEC.loader.exec_module(DAG)


def scope_cases():
    """Return three invalid semantic cases that the current checker accepts."""
    history = DAG.Node(
        "attention_q0", "q0", 1, 0, 0, "attention", [], None,
        ["history:q_OTHER:r0:l0:t0:unregistered"], [], 0)
    future = DAG.Node(
        "embed_future_q0", "q0", 1, 0, 0, "embed", [], None, [], [], 0)
    wrong_gate = DAG.Node(
        "gate_q_other", "q_OTHER", 0, 1, 0, "gate", [], None, [], [], 0)
    continuation = DAG.Node(
        "continue_q0_r2", "q0", 0, 2, 0, "next_pass",
        [wrong_gate.node_id], None, [], [], 0,
        metadata={"requires_online_gate": True})
    return [
        {
            "id": "unregistered_external_history",
            "nodes": [history],
            "expected_semantic_legal": False,
            "requirement": (
                "An external history read needs a registered committed version. "
                "Its request, generation, layer, depth, and position must match "
                "the consumer. A history: prefix alone is not that evidence."),
            "missing_witness": "Registered committed history for request q0.",
        },
        {
            "id": "future_token_without_sampling_edge",
            "nodes": [future],
            "expected_semantic_legal": False,
            "requirement": (
                "For non-prefill, non-speculative decode, input token 1 needs "
                "its actual sampled token ID from output position 0. "
                "A predecessor or registered prior completion must supply it."),
            "missing_witness": "Token-ID producer or prior completion for q0 token 1.",
        },
        {
            "id": "wrong_request_gate",
            "nodes": [wrong_gate, continuation],
            "expected_semantic_legal": False,
            "requirement": (
                "A guarded R3 continuation needs the R2 continuation decision "
                "for its own request, token, generation, and state version. "
                "Another request's gate cannot supply that decision. "
                "The original policy also needs its ordered hazard state."),
            "missing_witness": "Matching q0 token0 R2 continuation decision.",
        },
    ]


class ScopeCharacterizationTests(unittest.TestCase):
    """Assert observed acceptance, not correctness of the invalid examples."""

    def test_current_checker_accepts_three_semantically_invalid_cases(self):
        cases = scope_cases()
        self.assertEqual(len(cases), 3)
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertFalse(case["expected_semantic_legal"])
                self.assertTrue(case["requirement"])
                self.assertTrue(case["missing_witness"])
                self.assertTrue(DAG.validate(case["nodes"]))

    def test_history_example_has_no_registration_or_producer(self):
        case = scope_cases()[0]
        node = case["nodes"][0]
        self.assertEqual(node.request_id, "q0")
        self.assertIn("q_OTHER", node.kv_reads[0])
        self.assertEqual(node.predecessors, [])
        self.assertFalse(any(n.kv_writes for n in case["nodes"]))

    def test_future_example_is_decode_not_known_prefill_or_speculation(self):
        node = scope_cases()[1]["nodes"][0]
        self.assertEqual(node.token_or_chunk_id, 1)
        self.assertEqual(node.predecessors, [])
        self.assertEqual(node.scratch_version, "committed")
        self.assertNotIn("known_input_token_id", node.metadata)

    def test_gate_example_isolates_request_identity(self):
        gate, consumer = scope_cases()[2]["nodes"]
        self.assertEqual(gate.token_or_chunk_id, consumer.token_or_chunk_id)
        self.assertEqual(gate.r + 1, consumer.r)
        self.assertNotEqual(gate.request_id, consumer.request_id)
        self.assertEqual(consumer.predecessors, [gate.node_id])

    def test_matching_gate_presence_positive_control(self):
        gate, consumer = scope_cases()[2]["nodes"]
        gate.request_id = consumer.request_id
        self.assertTrue(DAG.validate([gate, consumer]))
        # This checks the request mismatch only. It does not certify CDF policy.
        self.assertEqual(gate.request_id, consumer.request_id)


if __name__ == "__main__":
    unittest.main()
