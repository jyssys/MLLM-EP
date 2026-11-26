"""Regression checks for compact logical EC indexing and scheduler metadata."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from vllm.multimodal.inputs import PlaceholderRange

from .embedding_rows import SLICE_KEY, SlicedEncoderRows, suffix_rows
from ..mllm_epd_prefill_heavy_token_split.cuda_ipc_ec import CUDAIPCEncoderConnector, SliceAwareECMetadata
from ..mllm_epd_prefill_heavy_token_split.token_split import PLAN_KEY


class EmbeddingRowsTests(unittest.TestCase):
    def test_single_row_beyond_large_cut(self):
        span = PlaceholderRange(offset=15, length=7790)
        self.assertEqual(suffix_rows(span, 7804), (7789, 7790))

    def test_masked_span_maps_prompt_tokens_to_dense_rows(self):
        span = PlaceholderRange(offset=5, length=8,
                                is_embed=torch.tensor([True, False, True, False, False, True, True, False]))
        self.assertEqual(suffix_rows(span, 4), (0, 4))
        self.assertEqual(suffix_rows(span, 8), (2, 4))
        self.assertEqual(suffix_rows(span, 10), (2, 4))
        self.assertEqual(suffix_rows(span, 12), (4, 4))
        self.assertEqual(suffix_rows(span, 14), (4, 4))

    def test_compact_clone_supports_each_original_chunk_slice(self):
        original = torch.arange(32).reshape(8, 4)
        compact = original[5:8].clone()
        cached = SlicedEncoderRows(compact, 5)
        self.assertEqual(compact.untyped_storage().nbytes(), 3 * 4 * original.element_size())
        self.assertTrue(torch.equal(cached[5:6], original[5:6]))
        self.assertTrue(torch.equal(cached[6:8], original[6:8]))
        self.assertTrue(torch.equal(cached[5:8], original[5:8]))
        for unmatched in (slice(4, 6), slice(5, 9), slice(8, 6), slice(None, 8), slice(5, 8, 2), 6):
            with self.subTest(rows=unmatched):
                with self.assertRaises((TypeError, IndexError)):
                    cached[unmatched]

    def make_scheduler(self):
        connector = CUDAIPCEncoderConnector.__new__(CUDAIPCEncoderConnector)
        connector.slice_aware = True
        connector._mm_datas_need_loads = {}
        connector._slice_rows_by_key = {}
        return connector

    @staticmethod
    def request(**options):
        params = {PLAN_KEY: {"prefix_tokens": 7804, "full_prompt_tokens": 7823},
                  "do_remote_prefill": False, "_remote_blocks_processed": True,
                  "remote_block_ids": [[0]], "remote_num_tokens": 7804}
        params.update(options)
        return SimpleNamespace(
            kv_transfer_params=params, num_computed_tokens=7804, num_prompt_tokens=7823,
            mm_features=[SimpleNamespace(identifier="request_unique_media",
                                         mm_position=PlaceholderRange(offset=15, length=7790))],
        )

    @staticmethod
    def allocated(scheduler, request, index):
        scheduler._mm_datas_need_loads[request.mm_features[index].identifier] = 7790

    def test_metadata_preserves_cut_after_nixl_clears_remote_prefill_flag(self):
        connector = self.make_scheduler()
        parent = "poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.ECExampleConnector"
        with patch(parent + ".update_state_after_alloc", new=self.allocated):
            connector.update_state_after_alloc(self.request(), 0)
        metadata = connector.build_connector_meta(None)
        self.assertIsInstance(metadata, SliceAwareECMetadata)
        self.assertEqual(metadata.rows_by_key, {"request_unique_media": (7789, 7790)})
        self.assertEqual(metadata.mm_datas[0].num_token, 7790)
        self.assertFalse(connector.build_connector_meta(None).mm_datas)

    def test_same_step_conflict_and_whole_arm_keep_full_rows(self):
        connector = self.make_scheduler()
        parent = "poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.ECExampleConnector"
        with patch(parent + ".update_state_after_alloc", new=self.allocated):
            connector.update_state_after_alloc(self.request(), 0)
            connector.update_state_after_alloc(self.request(**{SLICE_KEY: False}), 0)
        metadata = connector.build_connector_meta(None)
        self.assertNotIsInstance(metadata, SliceAwareECMetadata)
        self.assertEqual(metadata.mm_datas[0].num_token, 7790)

    def test_unproven_import_retains_whole_embedding(self):
        connector = self.make_scheduler()
        parent = "poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.ECExampleConnector"
        request = self.request()
        request.num_computed_tokens = 7803
        with patch(parent + ".update_state_after_alloc", new=self.allocated):
            connector.update_state_after_alloc(request, 0)
        self.assertNotIsInstance(connector.build_connector_meta(None), SliceAwareECMetadata)

    def test_same_media_in_two_occurrences_has_distinct_cache_identifiers(self):
        from ..mllm_epd_prefill_heavy_token_split.client import messages

        row = {"request_id": "same-frozen-input", "question": "Read the page.",
               "images": [{"path": "/unchanged/input.png"}]}
        first, second = [messages(row, nonce)[0]["content"][0] for nonce in ("trial_0000", "trial_0001")]
        self.assertEqual(first["image_url"], second["image_url"])
        self.assertNotEqual(first["uuid"], second["uuid"])


if __name__ == "__main__":
    unittest.main()
