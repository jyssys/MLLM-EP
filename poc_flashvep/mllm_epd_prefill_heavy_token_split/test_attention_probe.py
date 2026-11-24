"""Keep the optional diagnostic limited to the first LLM prefill layer."""
import unittest

from .forward_probe import capture_attention_allowed


class AttentionProbeTest(unittest.TestCase):
    def test_first_layer_prefill_is_selected(self):
        self.assertTrue(capture_attention_allowed('language_model.model.layers.0.self_attn.attn',
            dict(index=0,query_tokens=2416),4))

    def test_other_layers_and_decode_are_not_selected(self):
        for name in ['language_model.model.layers.10.self_attn.attn',
                     'language_model.model.layers.1.self_attn.attn','visual.blocks.0.attn']:
            self.assertFalse(capture_attention_allowed(name,dict(index=0,query_tokens=2416),4))
        self.assertFalse(capture_attention_allowed('model.layers.0.self_attn.attn',
            dict(index=0,query_tokens=1),4))
        self.assertFalse(capture_attention_allowed('model.layers.0.self_attn.attn',None,4))

    def test_capture_limit_is_enforced(self):
        self.assertFalse(capture_attention_allowed('model.layers.0.self_attn.attn',
            dict(index=4,query_tokens=2416),4))


if __name__=='__main__':
    unittest.main()
