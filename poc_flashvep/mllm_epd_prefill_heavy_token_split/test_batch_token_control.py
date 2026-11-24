"""Check real prepare/finish protocol helpers without a GPU or HTTP service."""
import unittest
from unittest.mock import AsyncMock, patch

from . import batch_token_correctness_control as control


class BatchTokenControlTest(unittest.IsolatedAsyncioTestCase):
    async def test_original_media_and_exact_handoff_survive_both_stages(self):
        body=dict(messages=[dict(role='user',content=['original_image','original_question'])],
            max_tokens=2048,ignore_eos=False,stream=False)
        row=dict(request_id='source_id',total_input_tokens=1000,images=[{},{}])
        params=dict(remote_num_tokens=800,remote_block_ids=[[1,2]],
            do_remote_prefill=True,do_remote_decode=False,remote_engine_id='actual_nixl_engine')
        outputs=[dict(usage=dict(prompt_tokens=1000)),dict(usage=dict(prompt_tokens=800),kv_transfer_params=params),
                 dict(usage=dict(prompt_tokens=1000,completion_tokens=1),choices=[dict(token_ids=[42])]),
                 [dict(released=True)]]
        post=AsyncMock(side_effect=outputs)
        with patch.object(control,'post',post),patch.object(control,'payload',return_value=body):
            prepared=await control.prepare('session',row,'actual_base',800)
            value=await control.finish('session',prepared)
        self.assertEqual(value['response']['choices'][0]['token_ids'],[42])
        self.assertEqual([call.args[1] for call in post.call_args_list],[18840,18841,18842,18840])
        for call in post.call_args_list[:3]:
            request=call.args[3]
            self.assertEqual(request['messages'],body['messages'])
            self.assertEqual(request['max_tokens'],1)
            self.assertFalse(request['ignore_eos'])
        destination=post.call_args_list[2].args[3]
        self.assertEqual(destination['kv_transfer_params']['remote_num_tokens'],800)
        self.assertEqual(destination['kv_transfer_params']['remote_engine_id'],'actual_nixl_engine')
        self.assertEqual(destination['kv_transfer_params'][control.PLAN_KEY],dict(prefix_tokens=800,full_prompt_tokens=1000))
        self.assertEqual(body['max_tokens'],2048,'Do not mutate the shared original request')


if __name__=='__main__':unittest.main()
