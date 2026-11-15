"""Check bytes per token before comparing payloads with telemetry."""
import unittest
from .geometry_transport_audit import structural_KV_bytes


class TransportAccountingTests(unittest.TestCase):
    def test_qwen_eight_head_KV_geometry(self):
        self.assertEqual(structural_KV_bytes(1,1),4096)
        self.assertEqual(structural_KV_bytes(8000,36),1125*1024**2)
        self.assertEqual(structural_KV_bytes(8000,24),750*1024**2)
    def test_hidden_is_separate_from_partial_KV(self):
        hidden=8000*4096*2*2
        self.assertEqual(hidden,125*1024**2)
        self.assertEqual(structural_KV_bytes(8000,24)+hidden,875*1024**2)


if __name__=='__main__':unittest.main()
