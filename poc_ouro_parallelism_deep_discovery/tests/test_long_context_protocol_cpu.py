import unittest

from poc_ouro_parallelism_deep_discovery.contract import ROOT, MODELS, read


class LongContextProtocolTests(unittest.TestCase):
    def test_native_pool_including_null_block(self):
        protocol = read(ROOT / 'artifacts/native_long_context_protocol_v1.json')
        for size, model in MODELS.items():
            for tp in [1, 2, 4]:
                for context in protocol['contexts']:
                    with self.subTest(size=size, tp=tp, context=context):
                        blocks = (context + protocol['generated_tokens'] + 15) // 16
                        per_block = 2 * model['layers'] * 4 * 16 * (16 // tp) * 128 * 2
                        required = (blocks + 1) * per_block
                        self.assertLessEqual(required, protocol['kv_pool_aggregate_bytes'] // tp)
                        self.assertLessEqual(context + protocol['generated_tokens'], protocol['max_model_len'])
        self.assertGreaterEqual(protocol['max_num_batched_tokens'], protocol['max_model_len'])

    def test_frozen_work_and_model_context(self):
        protocol = read(ROOT / 'artifacts/native_long_context_protocol_v1.json')
        self.assertEqual(protocol['status'], 'FROZEN_BEFORE_RESULTS')
        self.assertEqual(protocol['generated_tokens'], 128)
        self.assertEqual(protocol['max_num_seqs'], 1)
        self.assertEqual(protocol['cudagraph_capture_sizes'], [1])
        for model in MODELS.values():
            config = read(model['path'] / 'config.json')
            self.assertLessEqual(protocol['max_model_len'], config['max_position_embeddings'])
            self.assertIsNone(config['rope_scaling'])
            self.assertEqual(config['num_attention_heads'], 16)
            self.assertEqual(config['num_key_value_heads'], 16)


if __name__ == '__main__':
    unittest.main()
