from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from . import stable_ec as cache
from .stable_ec_server import occurrence_request, with_image_keys


class Storage:
    def __init__(self, pointer, size):
        self.pointer = pointer
        self.size = size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class Tensor:
    shape = (4, 3584)
    dtype = "bfloat16"

    def __init__(self, storage):
        self.storage = storage
        self.copy_ = Mock()

    def untyped_storage(self):
        return self.storage

    def numel(self):
        return self.storage.nbytes()

    def element_size(self):
        return 1


OWNER1, OWNER2, OWNER3 = (char * 64 for char in "012")
IMAGE1, IMAGE2, IMAGE3 = (char * 64 for char in "abc")


def store(registry, key, tensor):
    cache.ipc.write(cache.ipc.directory(registry.root, key) / "cuda_ipc_ready.json", {"key": key})
    registry.held[key] = tensor
    registry.ready[key] = object()
    registry.saved_count += 1


class StableTest(TestCase):
    def setUp(self):
        self.scratch = TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(cache.ipc.ProducerRegistry, "save", store).start()

    def registry(self, capacity=200, retention=100, reuse=True):
        return cache.StableProducerRegistry(Path(self.scratch.name), capacity, retention, reuse)

    def test_repeat_ownership_shares_source_without_replacing_original(self):
        registry = self.registry()
        self.assertEqual(registry.acquire(OWNER1, [IMAGE1, IMAGE1])["hits"], [])
        tensor = Tensor(Storage(10, 60))
        registry.save(cache.source_alias(OWNER1, IMAGE1), tensor)
        self.assertEqual(registry.acquire(OWNER2, [IMAGE1])["hits"], [IMAGE1])
        alias = cache.source_alias(OWNER2, IMAGE1)
        self.assertTrue((registry.root / alias / cache.ALIAS_NAME).exists())
        registry.save(alias, Tensor(Storage(20, 60)))
        self.assertIs(registry.held[IMAGE1], tensor)
        state = registry.release_owner(OWNER1)
        self.assertEqual(state["active_request_leases"], 1)
        self.assertEqual(state["idle_images"], 0)
        state = registry.release_owner(OWNER2)
        self.assertFalse((registry.root / alias).exists())
        self.assertEqual(state["idle_storage_bytes"], 60)
        self.assertEqual(state["acquired_image_hits"], 1)
        self.assertEqual(state["acquired_image_misses"], 1)

    def test_cacheless_mode_keeps_export_until_completion_ack(self):
        registry = self.registry(retention=0, reuse=False)
        registry.acquire(OWNER1, [IMAGE1])
        registry.save(cache.source_alias(OWNER1, IMAGE1), Tensor(Storage(10, 70)))
        self.assertEqual(registry.acquire(OWNER2, [IMAGE1])["hits"], [])
        registry.exports["test"] = {"keys": {IMAGE1}, "consumer": "prefill"}
        registry.release_owner(OWNER1)
        registry.release_owner(OWNER2)
        self.assertEqual(registry.state()["live_storage_bytes"], 70)
        self.assertEqual(registry.state()["idle_images"], 0)
        self.assertTrue(registry.acknowledge("test")["acknowledged"])
        self.assertEqual(registry.state()["live_storage_bytes"], 0)
        self.assertFalse((registry.root / IMAGE1).exists())
        registry.save(cache.source_alias(OWNER1, IMAGE1), Tensor(Storage(20, 70)))
        self.assertEqual(registry.state()["late_image_saves"], 1)

    def test_real_byte_limit_evicts_only_idle_storage_and_deduplicates_views(self):
        registry = self.registry(capacity=100, retention=100)
        first = Storage(10, 60)
        registry.acquire(OWNER1, [IMAGE1, IMAGE2])
        registry.save(cache.source_alias(OWNER1, IMAGE1), Tensor(first))
        registry.save(cache.source_alias(OWNER1, IMAGE2), Tensor(first))
        self.assertEqual(registry.state()["live_storage_bytes"], 60)
        registry.acquire(OWNER2, [IMAGE3])
        with self.assertRaises(MemoryError):
            registry.save(cache.source_alias(OWNER2, IMAGE3), Tensor(Storage(20, 60)))
        self.assertEqual(registry.state()["live_storage_bytes"], 60)
        registry.release_owner(OWNER1)
        registry.save(cache.source_alias(OWNER2, IMAGE3), Tensor(Storage(20, 60)))
        self.assertEqual(registry.state()["live_storage_bytes"], 60)
        self.assertEqual(set(registry.held), {IMAGE3})
        self.assertEqual(registry.state()["cache_evictions"], 2)

    def test_pinned_tier_publishes_then_releases_active_copy_and_evicts_oldest(self):
        registry = cache.HostRetainedProducerRegistry(Path(self.scratch.name), 200, 100, True)

        def stage(tensor):
            return Tensor(Storage(tensor.storage.pointer + 100, tensor.storage.size))

        with patch.object(registry, "_allocate_host", side_effect=lambda key: stage(registry.held[key])), \
                patch.object(registry, "_to_gpu", side_effect=stage), \
                patch("torch.cuda.Stream"), patch("torch.cuda.stream", return_value=nullcontext()):
            registry.acquire(OWNER1, [IMAGE1])
            registry.save(cache.source_alias(OWNER1, IMAGE1), Tensor(Storage(10, 60)))
            first = registry.release_owner(OWNER1)
            self.assertEqual(first["host_store_bytes"], 60)
            self.assertEqual(first["live_storage_bytes"], 0)
            self.assertEqual(registry.acquire(OWNER2, [IMAGE1])["hits"], [IMAGE1])
            self.assertIn(IMAGE1, registry.held)
            warm = registry.release_owner(OWNER2)
            self.assertEqual(warm["host_load_bytes"], 60)
            self.assertEqual(warm["host_store_bytes"], 60)
            self.assertEqual(warm["computed_image_saves"], 1)
            self.assertEqual(warm["live_storage_bytes"], 0)
            registry.acquire(OWNER3, [IMAGE2])
            registry.save(cache.source_alias(OWNER3, IMAGE2), Tensor(Storage(20, 70)))
            final = registry.release_owner(OWNER3)
            self.assertEqual(final["host_evictions"], 1)
            self.assertEqual(final["host_retained_storage_bytes"], 70)
            self.assertEqual(set(registry.host), {IMAGE2})

    def test_pinned_batch_fences_surviving_copies_once_before_publication(self):
        registry = cache.HostRetainedProducerRegistry(Path(self.scratch.name), 200, 90, True)
        registry.acquire(OWNER1, [IMAGE1, IMAGE2, IMAGE3])
        tensors = {key: Tensor(Storage(pointer, size)) for key, pointer, size in
                   ((IMAGE1, 10, 40), (IMAGE2, 20, 40), (IMAGE3, 30, 35))}
        for key, tensor in tensors.items():
            registry.save(cache.source_alias(OWNER1, key), tensor)

        def fence():
            self.assertEqual(registry.state()["host_pending_storage_bytes"], 75)
            self.assertEqual(registry.state()["host_retained_storage_bytes"], 0)
            self.assertEqual(set(registry.held), set(tensors))
            self.assertEqual(set(registry._pending_host), {IMAGE2, IMAGE3})
            for key in (IMAGE2, IMAGE3):
                registry._pending_host[key].copy_.assert_called_once_with(tensors[key], non_blocking=True)

        stream = Mock()
        stream.synchronize.side_effect = fence
        with patch.object(registry, "_allocate_host", side_effect=lambda key: Tensor(tensors[key].storage)), \
                patch("torch.cuda.Stream", return_value=stream), patch("torch.cuda.stream", return_value=nullcontext()):
            state = registry.release_owner(OWNER1)
        self.assertEqual(stream.wait_event.call_count, 2)
        stream.synchronize.assert_called_once()
        self.assertEqual(state["host_store_batches"], 1)
        self.assertEqual(state["host_stores"], 2)
        self.assertEqual(state["host_peak_storage_bytes"], 75)
        self.assertEqual(state["host_pending_storage_bytes"], 0)
        self.assertEqual(state["live_storage_bytes"], 0)

    def test_failed_host_copy_fence_keeps_both_endpoints_and_prevents_eviction(self):
        registry = cache.HostRetainedProducerRegistry(Path(self.scratch.name), 200, 100, True)
        registry.acquire(OWNER1, [IMAGE1])
        registry.save(cache.source_alias(OWNER1, IMAGE1), Tensor(Storage(10, 60)))
        stream = Mock()
        stream.synchronize.side_effect = RuntimeError("Copy could not be fenced")
        with patch.object(registry, "_allocate_host", return_value=Tensor(Storage(20, 60))), \
                patch("torch.cuda.Stream", return_value=stream), patch("torch.cuda.stream", return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "could not be fenced"):
                registry.release_owner(OWNER1)
        self.assertEqual(registry.state()["host_pending_storage_bytes"], 60)
        self.assertEqual(registry.state()["host_retained_storage_bytes"], 0)
        self.assertEqual(registry.state()["live_storage_bytes"], 60)
        with self.assertRaisesRegex(RuntimeError, "before its host copy is fenced"):
            registry._discard(IMAGE1)

    def test_input_layer_separates_occurrence_identity_from_reused_source(self):
        original = ChatCompletionRequest(messages=[{
            "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "file:///unused.png"}, "uuid": IMAGE1},
                {"type": "text", "text": "Compare the two inputs."},
                {"type": "image_url", "image_url": {"url": "file:///unused.png"}, "uuid": IMAGE1},
            ],
        }])
        request, identities = occurrence_request(original, OWNER1)
        self.assertEqual(identities, [IMAGE1, IMAGE1])
        originals = original.messages[0]["content"]
        occurrences = request.messages[0]["content"]
        self.assertEqual(originals[1], occurrences[1])
        self.assertEqual(originals[0]["uuid"], originals[2]["uuid"])
        self.assertNotEqual(occurrences[0]["uuid"], occurrences[2]["uuid"])
        source = object()
        prompt = {"type": "multimodal", "mm_hashes": {"image": ["first", "second"]},
                  "mm_kwargs": {"image": [source, source]}}
        _, stable = with_image_keys(([], [prompt]), identities)
        self.assertIs(stable[0]["mm_kwargs"]["image"][0], source)
        self.assertEqual(stable[0]["mm_hashes"]["image"], identities)
        self.assertEqual(prompt["mm_hashes"]["image"], ["first", "second"])
