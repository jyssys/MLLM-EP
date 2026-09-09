"""Exercise the official Req.check_finished seam, not a reimplemented rule."""
import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req, FINISH_MATCHED_TOKEN
from sglang.srt.sampling.sampling_params import SamplingParams


class EOSTermination(unittest.TestCase):
    def test_eos_after_non_eos_tokens(self):
        params = SamplingParams(temperature=0, max_new_tokens=16, ignore_eos=False)
        params.normalize(None)
        req = Req("regression", "fixture", [10, 11], params)
        req.tokenizer = SimpleNamespace(eos_token_id=151645, additional_stop_token_ids=None)
        req.output_ids.extend([19, 17])
        req.check_finished()
        self.assertIsNone(req.finished_reason)
        req.output_ids.append(151645)
        req.check_finished()
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)

    def test_ignore_eos_preserves_fixed_output_benchmark(self):
        params = SamplingParams(temperature=0, max_new_tokens=16, ignore_eos=True)
        params.normalize(None)
        req = Req("regression", "fixture", [10, 11], params)
        req.tokenizer = SimpleNamespace(eos_token_id=151645, additional_stop_token_ids=None)
        req.output_ids.extend([19, 17, 151645])
        req.check_finished()
        self.assertIsNone(req.finished_reason)


if __name__ == "__main__":
    unittest.main()
