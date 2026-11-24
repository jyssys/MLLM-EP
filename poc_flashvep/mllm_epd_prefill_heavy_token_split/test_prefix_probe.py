"""Probe ownership must preserve metadata and reject serving-answer claims."""
from types import SimpleNamespace
import unittest

from .prefix_probe_contract import KEY,truncate_probe


class PrefixProbeTest(unittest.TestCase):
    def request(self,params=None,max_tokens=1):
        return SimpleNamespace(prompt_token_ids=list(range(2961)),mm_features=[object(),object()],
            sampling_params=SimpleNamespace(extra_args={'kv_transfer_params':params or {}},max_tokens=max_tokens))

    def test_unmarked_request_is_unchanged(self):
        row=self.request();tokens=row.prompt_token_ids
        self.assertIs(truncate_probe(row,'pd'),row);self.assertIs(row.prompt_token_ids,tokens)

    def test_valid_real_prefix_retains_all_original_image_features(self):
        row=self.request({KEY:dict(prefix_tokens=128,full_prompt_tokens=2961)})
        features=row.mm_features;tokens=row.prompt_token_ids
        truncate_probe(row,'pd')
        self.assertEqual(row.prompt_token_ids,tokens[:128]);self.assertEqual(len(tokens),2961)
        self.assertIs(row.mm_features,features)

    def test_wrong_role_sampling_or_remote_contract_is_rejected(self):
        plan={KEY:dict(prefix_tokens=128,full_prompt_tokens=2961)}
        with self.assertRaises(AssertionError):truncate_probe(self.request(plan),'decode')
        with self.assertRaises(AssertionError):truncate_probe(self.request(plan,max_tokens=16),'pd')
        with self.assertRaises(AssertionError):truncate_probe(self.request(dict(plan,do_remote_decode=True)),'pd')

    def test_zero_full_or_wrong_original_length_is_rejected(self):
        for prefix,full in [(0,2961),(2961,2961),(128,2960)]:
            with self.assertRaises(AssertionError):
                truncate_probe(self.request({KEY:dict(prefix_tokens=prefix,full_prompt_tokens=full)}),'pd')


if __name__=='__main__':unittest.main()
