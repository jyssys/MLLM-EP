"""CPU-only control-flow test for optional request/event association."""
import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mechanism_probe import install, batch_identity


class FakeEvent:
    def __init__(self, enable_timing):
        assert enable_timing
    def record(self):
        pass
    def query(self):
        return True
    def elapsed_time(self, other):
        return 1.25


class FixtureDecoderLayer:
    def forward(self, value):
        return value + 1


class AssociationTest(unittest.TestCase):
    def test_request_shape_key_ignores_local_counter_but_not_position(self):
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid="r", output_ids=[1])],
                                input_ids=torch.zeros(1, dtype=torch.int64),
                                forward_mode="DECODE", prefix_lens=None,
                                extend_lens=None, seq_lens_sum=1025)
        first = batch_identity(batch)
        batch.local_invocation_id = 999
        self.assertEqual(first, batch_identity(batch))
        batch.reqs[0].output_ids.append(2)
        self.assertNotEqual(first, batch_identity(batch))
        self.assertIsNone(batch_identity(None))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_alp_unprofiled_startup_is_unchanged(self):
        source = Path(os.environ["SUCCESSOR_FASTPP_SOURCE"]) / "python/sglang/srt/managers/alp_scheduler.py"
        spec = importlib.util.spec_from_file_location("official_alp_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="fastpp-startup-probe-test-") as tmp:
            alp = module.ALPScheduler(candidate_chunks=[128], coeff_path=tmp + "/uncreated.json")
            # This is the real official unprofiled path: learning is a no-op.
            alp.update_runtime_overhead(.01, 128)
            self.assertEqual(alp.interference_updates, 0)
            runner = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: []),
                                     forward=lambda value: value)
            sched = SimpleNamespace(pp_rank=0, tp_rank=0, device="cuda:0",
                                    tp_worker=SimpleNamespace(model_runner=runner),
                                    alp_scheduler=alp, run_batch=lambda batch, pipe: None)
            with patch.dict(os.environ, {"SUCCESSOR_FASTPP_TRACE": tmp,
                                         "CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
                install(sched)
                alp.update_runtime_overhead(.01, 128)
                self.assertEqual(alp.interference_updates, 0)
                rows = [json.loads(s) for s in next(Path(tmp).glob("*.jsonl")).read_text().splitlines()]
                learned = next(r for r in rows if r["kind"] == "alp_learning_sample")
                self.assertIsNone(learned["predicted_before_update_host_s"])
                self.assertFalse(torch.cuda.is_initialized())

    def test_alp_observation_preserves_prediction_and_update(self):
        class Predictor:
            prefill_exec_time_by_chunk = {128: .01}
            bias = 0.0

            def predict_exec_time_for_chunk(self, chunk_size, **features):
                return self.prefill_exec_time_by_chunk[chunk_size] + self.bias

            def update_runtime_overhead(self, observed_exec_time, runtime_chunk_size=None,
                                        **features):
                self.bias = observed_exec_time - self.prefill_exec_time_by_chunk[runtime_chunk_size]

        alp = Predictor()
        runner = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: []),
                                 forward=lambda value: value)
        sched = SimpleNamespace(pp_rank=0, tp_rank=0, device="cuda:0",
                                tp_worker=SimpleNamespace(model_runner=runner),
                                alp_scheduler=alp, run_batch=lambda batch, pipe: None)
        with tempfile.TemporaryDirectory(prefix="fastpp-alp-probe-test-") as tmp:
            with patch.dict(os.environ, {"SUCCESSOR_FASTPP_TRACE": tmp,
                                         "CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
                install(sched)
                self.assertEqual(alp.predict_exec_time_for_chunk(128, decode_ctx_sum=512), .01)
                alp.update_runtime_overhead(.02, 128, decode_ctx_sum=512)
                self.assertEqual(alp.bias, .01)
                self.assertEqual(alp.predict_exec_time_for_chunk(128), .02)
                rows = [json.loads(s) for s in next(Path(tmp).glob("*.jsonl")).read_text().splitlines()]
                learned = next(r for r in rows if r["kind"] == "alp_learning_sample")
                self.assertEqual(learned["predicted_before_update_host_s"], .01)
                self.assertEqual(learned["observed_host_s"], .02)
                self.assertEqual(learned["features"]["decode_ctx_sum"], 512)
                self.assertFalse(torch.cuda.is_initialized())

    def test_preserves_result_and_request_identity(self):
        layer = FixtureDecoderLayer()
        runner = SimpleNamespace(
            model=SimpleNamespace(named_modules=lambda: [("model.layers.16", layer)]),
            forward=lambda value: layer.forward(value))
        sched = SimpleNamespace(pp_rank=1, tp_rank=0, device="cuda:1",
                                forward_ct=0, chunked_prefill_size=128,
                                cur_pp=SimpleNamespace(get=lambda: 2),
                                tp_worker=SimpleNamespace(model_runner=runner))
        sched.run_batch = lambda batch, pipe: runner.forward(41)
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid="run:request", output_ids=[2])],
                                input_ids=torch.zeros(12, dtype=torch.int64),
                                forward_mode="DECODE")
        with tempfile.TemporaryDirectory(prefix="fastpp-probe-test-") as tmp:
            with patch.dict(os.environ, {"SUCCESSOR_FASTPP_TRACE": tmp,
                                         "CUDA_VISIBLE_DEVICES": "4,5,6,7"}), \
                    patch("torch.cuda.Event", FakeEvent):
                install(sched)
                self.assertEqual(sched.run_batch(batch, None), 42)
                rows = [json.loads(s) for s in next(Path(tmp).glob("*.jsonl")).read_text().splitlines()]
                stages = [r for r in rows if r["kind"] == "stage"]
                self.assertEqual(len(stages), 2)
                for row in stages:
                    self.assertEqual(row["request_ids"], ["run:request"])
                    self.assertEqual(row["local_invocation_id"], 1)
                    self.assertEqual(row["M"], 12)
                    self.assertEqual(row["cuda_ms"], 1.25)
                self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
