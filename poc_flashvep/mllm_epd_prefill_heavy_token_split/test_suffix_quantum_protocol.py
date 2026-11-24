"""The real isolated suffix controller must keep request IDs and ACK lifetimes aligned."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from .test_idle_proxy import Reply,Post
from .token_split import PLAN_KEY,POLICY_KEY


class Session:
    def __init__(self,length,quantum,wrong_prefix=False):
        self.length,self.quantum=length,quantum;self.wrong_prefix=wrong_prefix
        self.calls=[];self.D_completed=False

    def post(self,url,json,headers=None):
        self.calls.append((url,json,headers))
        async def result():
            if '/campaign/ec_ipc/release' in url:
                assert self.D_completed,'The original E cannot release before the D sample and clone completion'
                return Reply([dict(live_images=0,outstanding_exports=0)])
            if ':18840/' in url:return Reply(dict(usage=dict(prompt_tokens=self.length)))
            if ':18841/' in url:return Reply(dict(kv_transfer_params=dict(
                remote_num_tokens=self.length-self.quantum+int(self.wrong_prefix),remote_block_ids=[[1]],
                do_remote_prefill=True,do_remote_decode=False,remote_engine_id='CPU_PROTOCOL_FIXTURE')))
            assert ':18842/' in url and not json['stream']
            self.D_completed=True
            return Reply(dict(usage=dict(prompt_tokens=self.length,completion_tokens=1),choices=[dict(token_ids=[42])]))
        return Post(result)


class SuffixProtocolTest(unittest.IsolatedAsyncioTestCase):
    def row(self):return dict(request_id='original',total_input_tokens=3000,
        question='Original question',images=[dict(path='/tmp/unchanged_original_image.png')])

    async def test_full_prefix_import_and_same_metadata_identity_precede_release(self):
        from . import suffix_quantum_cost_control as module
        session=Session(3000,128)
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)):
            result=await module.trial(session,self.row(),128,'fixed_original')
        self.assertEqual(result['prefix_tokens'],2872)
        self.assertEqual(result['logical_prefix_KV_bytes'],2872*28*2*4*128*2)
        self.assertEqual(len(session.calls),4)
        encoder,source,destination,release=session.calls
        self.assertEqual(encoder[1]['kv_transfer_params'][POLICY_KEY]['fixed_prefix'],2872)
        self.assertEqual(source[1]['kv_transfer_params'][PLAN_KEY],destination[1]['kv_transfer_params'][PLAN_KEY])
        self.assertEqual(destination[2]['x-request-id'],'fixed_original:D')
        self.assertEqual(len({call[2]['x-campaign-metadata-key'] for call in session.calls[:3]}),1)
        self.assertFalse(result['task_completion'])

    async def test_invalid_remote_prefix_stops_before_decode_and_does_not_fake_an_ack(self):
        from . import suffix_quantum_cost_control as module
        session=Session(3000,128,wrong_prefix=True)
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)):
            with self.assertRaises(AssertionError):await module.trial(session,self.row(),128,'bad_prefix')
        self.assertEqual(len(session.calls),2)
        self.assertFalse(session.D_completed)


if __name__=='__main__':unittest.main()
