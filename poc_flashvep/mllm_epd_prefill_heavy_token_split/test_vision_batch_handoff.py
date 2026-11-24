"""Exercise the actual target-only diagnostic handoff, not a renderer identity stub."""
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from .test_metadata_handoff import rendered, request


class VisionBatchHandoffTest(unittest.TestCase):
    def test_joint_envelope_keeps_target_inputs_and_uses_live_encoder_identifiers(self):
        import torch
        from .processor_handoff import RenderStore
        from .vision_batch_handoff import prepare
        target=rendered()
        joint=copy.deepcopy(target)
        joint[1][0]['mm_hashes']['image']=['renderer-joint-target','renderer-joint-companion']
        joint[1][0]['mm_kwargs']['image']*=2
        joint[1][0]['mm_placeholders']['image']*=2
        with TemporaryDirectory() as folder:
            store=RenderStore(Path(folder));key='7'*64
            headers=prepare(store,target,request(),joint,key,9.)
            self.assertEqual(headers.get('x-campaign-metadata-key'),key,
                'P must consume the explicit original E identifiers, not rerender independent local UUIDs')
            self.assertEqual(headers['x-campaign-metadata-mode'],'full')
            actual=store.load(key,request(),arrival_time=9.)
            self.assertEqual(actual[1][0]['prompt_token_ids'],target[1][0]['prompt_token_ids'])
            self.assertEqual(actual[1][0]['mm_placeholders'],target[1][0]['mm_placeholders'])
            self.assertEqual(actual[1][0]['mm_hashes']['image'],['renderer-joint-target'])
            self.assertEqual(len(actual[1][0]['mm_kwargs']['image']),1)
            self.assertEqual(target[1][0]['mm_hashes']['image'],['original-image'])
            for field in ['pixel_values','image_grid_thw']:
                self.assertTrue(torch.equal(actual[1][0]['mm_kwargs']['image'][0][field].data,
                    target[1][0]['mm_kwargs']['image'][0][field].data))


if __name__=='__main__':unittest.main()
