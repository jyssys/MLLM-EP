"""Focused CPU tests for the source-level DeepStack ubatch lifetime fix."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        deepstack_input_embeds=[torch.arange(6, dtype=torch.float32).view(6, 1)],
        deepstack_input_embeds_num_tokens=6,
        deepstack_input_embeds_dirty_tokens=6,
        deepstack_num_level=1,
    )


def _context(token_slice: slice) -> SimpleNamespace:
    return SimpleNamespace(additional_kwargs={"ubatch_token_slice": token_slice})


class DeepStackSourceFixTest(unittest.TestCase):
    def test_actual_token_slice_supports_more_than_two_consumers(self) -> None:
        model = _model()
        for token_slice, expected in (
            (slice(0, 2), [0.0, 1.0]),
            (slice(2, 4), [2.0, 3.0]),
            (slice(4, 6), [4.0, 5.0]),
        ):
            with patch(
                "vllm.model_executor.models.qwen3_vl.get_forward_context",
                return_value=_context(token_slice),
            ):
                result = Qwen3VLForConditionalGeneration._get_deepstack_input_embeds(
                    model, 2
                )
                self.assertIsNotNone(result)
                self.assertEqual(
                    result["deepstack_input_embeds_0"].flatten().tolist(), expected
                )
                Qwen3VLForConditionalGeneration._clear_deepstack_input_embeds(
                    model, 2
                )
                self.assertEqual(model.deepstack_input_embeds_num_tokens, 6)

    def test_exception_finalizes_wave_and_next_wave_has_no_stale_payload(
        self,
    ) -> None:
        model = _model()
        finalize_calls = 0

        def finalize() -> None:
            nonlocal finalize_calls
            finalize_calls += 1
            Qwen3VLForConditionalGeneration._finalize_ubatch_inputs(model)

        wrapper = object.__new__(UBatchWrapper)
        wrapper.runnable = SimpleNamespace(_finalize_ubatch_inputs=finalize)

        with self.assertRaisesRegex(RuntimeError, "forced consumer failure"):
            with wrapper._ubatch_input_lifetime():
                raise RuntimeError("forced consumer failure")

        self.assertEqual(finalize_calls, 1)
        self.assertEqual(model.deepstack_input_embeds_num_tokens, 0)
        with patch(
            "vllm.model_executor.models.qwen3_vl.get_forward_context",
            return_value=_context(slice(0, 2)),
        ):
            self.assertIsNone(
                Qwen3VLForConditionalGeneration._get_deepstack_input_embeds(model, 2)
            )

        next_payload = torch.tensor([[[10.0], [11.0], [12.0], [13.0]]])
        Qwen3VLForConditionalGeneration._set_deepstack_input_embeds(
            model, next_payload
        )
        self.assertEqual(model.deepstack_input_embeds_num_tokens, 4)
        self.assertEqual(
            model.deepstack_input_embeds[0].flatten().tolist(),
            [10.0, 11.0, 12.0, 13.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
