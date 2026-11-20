import unittest
from .serving import output_contract


class OutputReplayTests(unittest.TestCase):
    def test_natural_contract_is_unchanged(self):
        self.assertEqual(output_contract({},'decode','q0',2048,False),(2048,False))

    def test_only_measured_requests_are_overridden(self):
        cfg=dict(replay_output_lengths=[23],diagnostic_controlled_output_replay=True)
        self.assertEqual(output_contract(cfg,'decode','q0',2048,False),(23,True))
        self.assertEqual(output_contract(cfg,'prefill','q0',2048,False),(2048,True))
        self.assertEqual(output_contract(cfg,'decode','warm0',16,True),(16,True))

    def test_unlabeled_replay_is_rejected(self):
        with self.assertRaises(AssertionError):output_contract(dict(replay_output_lengths=[3]),'decode','q0',2048,False)


if __name__=='__main__':unittest.main()
