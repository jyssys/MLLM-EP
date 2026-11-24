"""Exercise the prefix controller policy against the real EC release gate."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import test_cuda_ipc_ec as fixture


class PrefixProbeLifetimeTest(unittest.TestCase):
    setUp=fixture.RegistryTest.setUp

    def test_only_needed_image_can_issue_a_descriptor_before_release(self):
        from .cuda_ipc_ec import ProducerRegistry
        controller=Path(__file__).with_name('prefix_quantum_correctness_control.py')
        calls=[node for node in ast.walk(ast.parse(controller.read_text()))
            if isinstance(node,ast.Call) and len(node.args)>1
            and isinstance(node.args[1],ast.Constant) and node.args[1].value=='ec_ipc/inline_policy']
        self.assertEqual(len(calls),1)
        policy={keyword.arg:ast.literal_eval(keyword.value) for keyword in calls[0].args[2].keywords}
        with TemporaryDirectory() as folder:
            registry=ProducerRegistry(folder)
            registry.inline_role=policy['consumer_role'] if policy['enabled'] else None
            fake=lambda tensor:(SimpleNamespace(__name__='rebuild_cuda_tensor'),fixture.reduction())
            with patch('torch.multiprocessing.reductions.reduce_tensor',side_effect=fake):
                registry.save('image-A',fixture.FakeTensor(100))
                registry.save('image-B',fixture.FakeTensor(200))
                # The original E encodes both images. A 128-token prefix uses A only.
                inline=Path(folder)/'image-A/cuda_ipc_inline_pd.json'
                descriptor=json.loads(inline.read_text()) if inline.exists() else registry.export(['image-A'],'pd')
            with self.assertRaisesRegex(AssertionError,'ACK is still pending'):
                registry.release(['image-A','image-B'])
            registry.acknowledge(descriptor['token'])
            result=registry.release(['image-A','image-B'])
            self.assertEqual(result['live_images'],0)
            self.assertEqual(result['outstanding_exports'],0)
            self.assertEqual(result['saved_count'],result['released_count'])


if __name__=='__main__':unittest.main()
