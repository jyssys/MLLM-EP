"""A split inside an image needs two independently acknowledged consumers."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def prompt():
    from vllm.multimodal.inputs import PlaceholderRange
    return dict(prompt_token_ids=list(range(12)), mm_hashes=dict(image=['image-A', 'image-B']),
                mm_placeholders=dict(image=[PlaceholderRange(2, 4), PlaceholderRange(8, 2)]))


def request(prefix):
    from .token_split import POLICY_KEY
    return SimpleNamespace(kv_transfer_params={POLICY_KEY: dict(suffix_fraction=0., fixed_prefix=prefix)})


class ConsumerPlanTest(unittest.TestCase):
    def test_boundaries_preserve_every_original_image(self):
        from .embedding_consumer_plan import build
        cases = [(2, [['decode'], ['decode']]),
                 (4, [['prefill', 'decode'], ['decode']]),
                 (6, [['prefill'], ['decode']]),
                 (8, [['prefill'], ['decode']]),
                 (11, [['prefill'], ['prefill']])]
        for prefix, expected in cases:
            with self.subTest(prefix=prefix):
                original = prompt()
                result = build(request(prefix), original)
                self.assertEqual([r['roles'] for r in result['images']], expected)
                self.assertEqual([r['key'] for r in result['images']], original['mm_hashes']['image'])
                self.assertEqual(result['prefix_tokens'], prefix)
                self.assertEqual(original['prompt_token_ids'], list(range(12)))

    def test_native_does_not_create_an_explicit_plan(self):
        from .embedding_consumer_plan import build
        self.assertIsNone(build(SimpleNamespace(kv_transfer_params=None), prompt()))

    def test_unqualified_noncontiguous_embedding_mask_fails(self):
        import torch
        from vllm.multimodal.inputs import PlaceholderRange
        from .embedding_consumer_plan import build
        value = prompt()
        value['mm_placeholders']['image'][0] = PlaceholderRange(2, 4,
            is_embed=torch.tensor([False, False, True, True]))
        with self.assertRaises(AssertionError):
            build(request(4), value)

    def test_plan_files_cannot_be_replaced(self):
        from .embedding_consumer_plan import build, persist
        with TemporaryDirectory() as root:
            plan = build(request(4), prompt())
            persist(root, 'request-key', plan)
            saved = json.loads((Path(root) / 'image-A/cuda_ipc_consumers.json').read_text())
            self.assertEqual(saved['roles'], ['prefill', 'decode'])
            with self.assertRaises(AssertionError):
                persist(root, 'another-request', plan)


class ConsumerLifetimeTest(unittest.TestCase):
    def test_dual_consumers_require_two_fresh_descriptors_and_two_acks(self):
        from .cuda_ipc_ec import ProducerRegistry
        from .test_cuda_ipc_ec import FakeTensor, reduction
        from .common import write
        from contextlib import nullcontext
        event = lambda: SimpleNamespace(record=lambda: None)
        stream = SimpleNamespace(wait_event=lambda _: None)
        fake = lambda _: (SimpleNamespace(__name__='rebuild_cuda_tensor'), reduction())
        with TemporaryDirectory() as root, patch('torch.cuda.Event', side_effect=event), \
                patch('torch.cuda.current_stream', return_value=stream), \
                patch('torch.cuda.Stream', return_value=object()), \
                patch('torch.cuda.stream', side_effect=lambda _: nullcontext()), \
                patch('torch.multiprocessing.reductions.reduce_tensor', side_effect=fake) as exported:
            folder = Path(root) / 'image-A'
            write(folder / 'cuda_ipc_consumers.json', dict(key='image-A', roles=['prefill', 'decode'],
                prefix_tokens=4, full_prompt_tokens=12))
            registry = ProducerRegistry(root)
            registry.inline_role = 'prefill'
            registry.save('image-A', FakeTensor())
            self.assertEqual(exported.call_count, 2)
            p = json.loads((folder / 'cuda_ipc_inline_prefill.json').read_text())
            d = json.loads((folder / 'cuda_ipc_inline_decode.json').read_text())
            self.assertNotEqual(p['token'], d['token'])
            registry.acknowledge(p['token'])
            with self.assertRaises(AssertionError):
                registry.release(['image-A'])
            registry.acknowledge(d['token'])
            self.assertEqual(registry.release(['image-A'])['live_images'], 0)


if __name__ == '__main__':
    unittest.main()
