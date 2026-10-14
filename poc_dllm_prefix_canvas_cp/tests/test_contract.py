import json
import unittest
from poc_dllm_prefix_canvas_cp.contract import *
from poc_dllm_prefix_canvas_cp.runtime import context_settings


class ContractTests(unittest.TestCase):
    def test_only_authorized_GPUs(self):self.assertEqual(MASK,'4,5,6,7')

    def test_context_and_native_mask_sizing(self):
        for p in PREFIXES:
            settings=context_settings(p)
            self.assertEqual(settings['logical_context'],p+64)
            self.assertEqual(settings['physical_key_capacity'],2*p)
            self.assertLessEqual(settings['physical_key_capacity'],MAX_CONTEXT)
        with self.assertRaises(AssertionError):context_settings(32768)

    def test_frozen_cohort_metadata_only(self):
        d=json.loads((ROOT/'results/frozen_manifest.json').read_text())
        self.assertEqual(sum(r['split']=='calibration' for r in d['requests']),8)
        self.assertEqual(sum(r['split']=='held_out' for r in d['requests']),24)
        self.assertEqual(d['token_artifact_sha256'],sha(ROOT/'results/frozen_tokens.json.gz'))

    def test_authoritative_spec(self):self.assertEqual(sha(SPEC),sha(ROOT/'authoritative_spec.md'))


if __name__=='__main__':unittest.main()
