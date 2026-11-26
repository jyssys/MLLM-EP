"""Check crossing descriptor choice, one-shot ownership, and retirement."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..mllm_epd_prefill_heavy_token_split.crossing_ipc import (
    CROSSING_PLAN, CROSSING_ROLE, cancellation_marker, claim_crossing_revoke, claim_inline,
    crossing_candidates, decode_inline_roles, inline_available, inline_paths,
)
from ..mllm_epd_prefill_heavy_token_split import cuda_ipc_ec as ipc


TOKEN = "a" * 32
REQUEST = "b" * 64


def add_inline(folder, key, role=CROSSING_ROLE, *, token=TOKEN):
    descriptor, taken = inline_paths(folder, role)
    descriptor.write_text(json.dumps({"consumer": role, "token": token, "items": [{"key": key}]}))
    return taken


class CrossingProtocolTests(unittest.TestCase):
    def test_envelope_selects_only_strictly_interior_cuts(self):
        prompt = {"prompt_token_ids": list(range(100)),
                  "mm_hashes": {"image": ["past", "one-row", "intersects", "later"]},
                  "mm_placeholders": {"image": [SimpleNamespace(offset=62, length=3),
                                                SimpleNamespace(offset=65, length=1),
                                                SimpleNamespace(offset=66, length=14),
                                                SimpleNamespace(offset=80, length=10)]}}
        self.assertEqual(crossing_candidates(prompt, .35, 0),
                         [("intersects", 66, 80), ("later", 80, 90)])
        self.assertEqual(crossing_candidates(prompt, .05, 36),
                         [("past", 62, 65), ("intersects", 66, 80), ("later", 80, 90)])
        self.assertEqual(crossing_candidates(prompt, .05, 0), [])

    def test_decode_prefers_unused_p_inline_before_spare(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            add_inline(folder, "image", "prefill")
            add_inline(folder, "image")
            self.assertEqual(decode_inline_roles(folder, True), ("prefill", CROSSING_ROLE))
            self.assertEqual(decode_inline_roles(folder, False), ("decode",))
            chosen = next(data for role in decode_inline_roles(folder, True)
                          if (data := claim_inline(folder, "image", role, "decode")) is not None)
            self.assertEqual(chosen["consumer"], "prefill")
            self.assertTrue(inline_available(folder, CROSSING_ROLE))
            (folder / "cuda_ipc_consumers.json").touch()
            self.assertEqual(decode_inline_roles(folder, True), ("decode",))

    def test_claim_and_revoke_are_mutually_exclusive(self):
        with TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=2) as workers:
            for index in range(16):
                folder = Path(directory) / str(index)
                folder.mkdir()
                taken = add_inline(folder, "image")
                start = Barrier(2)

                def claim():
                    start.wait()
                    return claim_inline(folder, "image", CROSSING_ROLE, "decode") is not None

                def revoke():
                    start.wait()
                    return claim_crossing_revoke(folder, TOKEN)

                consumer = workers.submit(claim)
                producer = workers.submit(revoke)
                claimed = consumer.result()
                self.assertNotEqual(claimed, producer.result())
                self.assertEqual(json.loads(taken.read_text())["disposition"], "claimed" if claimed else "revoked")
                self.assertFalse(claim_crossing_revoke(folder, TOKEN))
                self.assertIsNone(claim_inline(folder, "image", CROSSING_ROLE, "decode"))

    @staticmethod
    def registry(root):
        registry = ipc.ProducerRegistry(root)
        registry.crossing_preissue = True
        return registry

    @staticmethod
    def preissued(registry, key, token=TOKEN):
        folder = registry.root / key
        folder.mkdir()
        add_inline(folder, key, token=token)
        registry.held[key] = object()
        registry.ready[key] = object()
        registry.exports[token] = dict(keys={key}, consumer=CROSSING_ROLE, crossing_request=REQUEST,
                                       crossing_args={"ref": key})
        return folder

    def test_success_retires_an_unclaimed_reference_once(self):
        with TemporaryDirectory() as directory, patch.object(ipc, "release_unused_cuda_reference") as release:
            registry = self.registry(directory)
            folder = self.preissued(registry, "image")
            self.assertEqual(registry.release(["image"])["outstanding_exports"], 0)
            self.assertEqual(registry.release(["image"])["live_images"], 0)
            release.assert_called_once_with({"ref": "image"})
            self.assertEqual(registry.crossing_revoked, 1)
            self.assertFalse(inline_available(folder, CROSSING_ROLE))

    def test_claimed_reference_requires_real_ack_before_source_retirement(self):
        with TemporaryDirectory() as directory, patch.object(ipc, "release_unused_cuda_reference") as release:
            registry = self.registry(directory)
            folder = self.preissued(registry, "image")
            with self.assertRaises(FileNotFoundError):
                registry.acknowledge(TOKEN)
            with self.assertRaises(ValueError):
                claim_inline(folder, "image", CROSSING_ROLE, "prefill")
            self.assertIsNotNone(claim_inline(folder, "image", CROSSING_ROLE, "decode"))
            with self.assertRaisesRegex(AssertionError, "ACK is still pending"):
                registry.release(["image"])
            self.assertIn("image", registry.held)
            self.assertTrue(registry.acknowledge(TOKEN)["acknowledged"])
            self.assertEqual(registry.crossing_acknowledged, 1)
            self.assertEqual(registry.release(["image"])["outstanding_exports"], 0)
            release.assert_not_called()

    def test_cancel_retires_only_unused_spares_and_suppresses_late_preissue(self):
        with TemporaryDirectory() as directory, patch.object(ipc, "release_unused_cuda_reference") as release:
            registry = self.registry(directory)
            unused = self.preissued(registry, "unused")
            used = self.preissued(registry, "claimed", "c" * 32)
            self.assertIsNotNone(claim_inline(used, "claimed", CROSSING_ROLE, "decode"))
            state = registry.cancel_crossing(REQUEST)
            self.assertEqual((state["revoked"], state["still_claimed"]), (1, 1))
            self.assertTrue(cancellation_marker(directory, REQUEST).exists())
            self.assertFalse(inline_available(unused, CROSSING_ROLE))
            self.assertIn("unused", registry.held)
            self.assertIn("claimed", registry.held)
            release.assert_called_once_with({"ref": "unused"})
            later = registry.root / "later"
            later.mkdir()
            (later / CROSSING_PLAN).write_text(json.dumps({
                "key": "later", "request_key": REQUEST, "start": 70, "end": 90, "prompt_tokens": 100,
            }))
            with patch.object(registry, "export") as export:
                registry.preissue_crossing("later", later)
                export.assert_not_called()

    def test_retirement_uses_only_the_audited_ref_counter_slot(self):
        class Storage:
            calls = []

            @classmethod
            def _release_ipc_counter(cls, handle, offset, *, device):
                cls.calls.append((handle, offset, device))

        args = [None] * 15
        args[4], args[6], args[11], args[12] = Storage, 0, b"producer reference", 7
        with patch.object(ipc, "decode_reduction_args", return_value=args):
            ipc.release_unused_cuda_reference({"encoded": "descriptor"})
        self.assertEqual(Storage.calls, [(b"producer reference", 7, 0)])


if __name__ == "__main__":
    unittest.main()
