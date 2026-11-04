import unittest
from .work_accounting import recurrent_geometry, huginn_boundary_geometry, measured_partitions


class GeometryTests(unittest.TestCase):
    def test_warmup_partition_with_reused_seed_is_excluded(self):
        startup = dict(seed=7, time=9., request_id='startup')
        measured = dict(seed=7, time=11., request_id='measured')
        self.assertEqual(measured_partitions([startup, measured], {7}, 10.), {7: measured})

    def test_two_measured_partitions_still_fail(self):
        rows = [dict(seed=7, time=value, request_id=str(value)) for value in (11., 12.)]
        with self.assertRaisesRegex(AssertionError, 'actual measured boundary'):
            measured_partitions(rows, {7}, 10.)

    def test_equal_rows_are_not_equal_attention(self):
        loop = recurrent_geometry(4096, 4, 'loop', 3, 4096)
        token = recurrent_geometry(4096, 4, 'token', 4, 3072)
        self.assertEqual(loop['assigned_D_recurrent_token_rows'], token['assigned_D_recurrent_token_rows'])
        field = 'assigned_D_causal_attention_pairs_per_physical_attention_layer'
        self.assertGreater(token[field], loop[field])

    def test_full_and_local_endpoints(self):
        for axis, cut, prefix in (('loop', 4, 4096), ('token', 4, 4096)):
            value = recurrent_geometry(4096, 4, axis, cut, prefix)
            self.assertEqual(value['assigned_D_recurrent_token_rows'], 0)
        for axis, cut, prefix in (('loop', 0, 4096), ('token', 4, 0)):
            value = recurrent_geometry(4096, 4, axis, cut, prefix)
            self.assertEqual(value['assigned_D_recurrent_token_rows'], value['full_recurrent_token_rows'])

    def test_huginn_equal_core_rows_can_move_different_boundary_work(self):
        loop = huginn_boundary_geometry(2048, 32, 'loop', 31, 2048)
        token = huginn_boundary_geometry(2048, 32, 'token', 32, 1984)
        self.assertEqual(loop['D_recurrent_block_token_rows'], 8192)
        self.assertEqual(token['D_recurrent_block_token_rows'], 8192)
        self.assertEqual(loop['D_additional_prompt_Coda_KV_block_token_rows'], 4094)
        self.assertEqual(token['D_additional_prompt_Coda_KV_block_token_rows'], 126)
        self.assertEqual(token['D_additional_prelude_block_token_rows'], 128)


if __name__ == '__main__':
    unittest.main()
