"""The initial extension cut is not the Native request's actual Full36 cut."""
import unittest
from .configuration import base,state


class ConfigurationTest(unittest.TestCase):
    def test_native_initial_extension_cut_satisfies_the_real_constructor(self):
        for cfg in (state('Native',0,8),base('CPU_NATIVE_CONTRACT','Native',0,36,16,.5)):
            self.assertGreater(cfg['layer_cut'],0)
            self.assertLess(cfg['layer_cut'],36)
            self.assertEqual(cfg['policy'],'NATIVE')
            self.assertTrue(cfg['reuse_prefill_token'])

    def test_last2_exact_performance_rows(self):
        cfg=base('CPU_TOKEN_ROWS','Token',2,72,16,2.)
        self.assertEqual(sum(cfg['token_suffix_by_request'])*36,72*8000*2)


if __name__=='__main__':
    unittest.main()
