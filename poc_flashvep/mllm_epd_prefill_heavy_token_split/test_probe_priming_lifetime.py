"""Reach the actual inline-policy guard while a private primed E image remains live."""
import types
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from . import interference_probe_control as controller
from .test_idle_proxy import Post,Reply


class PrimingLifetimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_private_embedding_stays_live_without_global_policy_change_or_unused_export(self):
        state=dict(live_images=0,outstanding_exports=0)
        registry=types.SimpleNamespace(inline_role='prefill',state=lambda:state.copy())
        seen=[]
        class Session:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            def post(self,url,json,headers=None):
                seen.append((url,json,headers))
                async def result():
                    if ':18840/' in url:
                        state.update(live_images=2,outstanding_exports=2 if registry.inline_role=='prefill' else 0)
                    else:
                        assert ':18841/' in url
                        assert not json.get('kv_transfer_params'),'Priming must not create an unused P remote-KV reservation'
                        assert state['live_images']==2 and state['outstanding_exports']==2
                        state['outstanding_exports']=0
                    return Reply(dict(usage=dict(prompt_tokens=2961,completion_tokens=1),choices=[dict(token_ids=[42])]))
                return Post(result)
        def response(request,**kwargs):
            url=request if isinstance(request,str) else request.full_url
            method='GET' if isinstance(request,str) else request.get_method()
            self.assertTrue(url.endswith('/campaign/ec_ipc/state'))
            if method!='GET':raise HTTPError(url,405,'Method Not Allowed',{},None)
            return io.BytesIO(json.dumps([registry.state()]).encode())
        row=dict(total_input_tokens=2961,image_count=2)
        with TemporaryDirectory() as directory,patch.object(controller,'ROOT',Path(directory)),\
                patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.REGISTRY',registry),\
                patch.object(controller,'payload',return_value=dict(max_tokens=1,stream=False)),\
                patch.object(controller.urllib.request,'urlopen',side_effect=response),\
                patch('aiohttp.ClientSession',return_value=Session()):
            result=await controller.prime(row,'private_priming_fixture')
        self.assertEqual(result['full_tokens'],2961)
        self.assertEqual(registry.inline_role,'prefill')
        self.assertEqual(state,dict(live_images=2,outstanding_exports=0))
        self.assertEqual(len(seen),2)
        self.assertEqual(seen[0][2]['x-campaign-metadata-key'],seen[1][2]['x-campaign-metadata-key'])


if __name__=='__main__':unittest.main()
