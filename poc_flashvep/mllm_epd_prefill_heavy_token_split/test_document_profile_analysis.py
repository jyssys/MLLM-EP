"""Avoid interpreting dummy encoder KV or sampled states as continuous capacity."""
import unittest

from .document_stratum_analysis import resource_observations


class ResourceScopeTest(unittest.TestCase):
    def rows(self):
        return [dict(begin_ns=1000000000,before=dict(free_KV_blocks=4,total_KV_blocks=10,
            decode_active=2,decode_ready_waiting=1,waiting_status={'WAITING_FOR_REMOTE_KVS':3},
            encoder_free_slots=100)),dict(begin_ns=3000000000,before=dict(free_KV_blocks=9,total_KV_blocks=10,
            decode_active=0,decode_ready_waiting=0,waiting_status={},encoder_free_slots=300))]

    def test_encoder_dummy_pool_is_not_llm_headroom(self):
        result=resource_observations(self.rows(),0,4,'encoder')
        self.assertIsNone(result['free_KV_fraction_min'])
        self.assertIsNone(result['decode_active_max'])
        self.assertEqual(result['encoder_cache_free_slots_min'],100)

    def test_only_observations_inside_roi_count(self):
        result=resource_observations(self.rows(),0,2,'decode')
        self.assertEqual(result['observations'],1)
        self.assertEqual(result['free_KV_fraction_min'],.4)
        self.assertEqual(result['decode_active_max'],2)
        self.assertEqual(result['remote_KV_waiting_max'],3)
        self.assertIn('No interval interpolation',result['scope'])


if __name__=='__main__':unittest.main()
