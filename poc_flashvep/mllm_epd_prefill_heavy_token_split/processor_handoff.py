"""Exact, request-scoped CPU prompt handoff after online E preprocessing."""
import asyncio
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import time

import msgspec
from vllm.multimodal.inputs import MultiModalKwargsItem,MultiModalKwargsItems,PlaceholderRange
from vllm.v1.serial_utils import MsgpackEncoder,MsgpackDecoder

from .common import ROOT,append,digest,write


HANDOFF_CONTEXT=ContextVar('campaign_processor_handoff',default=None)
STAGE_ONLY_FIELDS={'max_tokens','max_completion_tokens','stream','stream_options','kv_transfer_params','request_id'}


class RenderBundle(msgspec.Struct):
    fingerprint:str
    mode:str
    conversation:list[dict]
    prompt:dict
    mm_kwargs:MultiModalKwargsItems
    mm_placeholders:dict[str,list[PlaceholderRange]]


def input_fingerprint(request):
    body=request.model_dump(mode='json')
    stable={key:value for key,value in body.items() if key not in STAGE_ONLY_FIELDS}
    return hashlib.sha256(json.dumps(stable,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


class RenderStore:
    def __init__(self,root):
        self.root=Path(root)

    def directory(self,key):
        assert re.fullmatch('[0-9a-f]{64}',key), 'Invalid request-scoped metadata key'
        target=self.root/key
        assert not target.is_symlink(), 'Metadata symlinks are forbidden'
        return target

    def save(self,key,request,rendered,mode):
        assert mode in ('full','grid_only')
        conversation,inputs=rendered
        assert len(inputs)==1 and inputs[0]['type']=='multimodal'
        from .embedding_consumer_plan import build,persist
        consumer_plan=build(request,inputs[0])
        if consumer_plan is not None:
            persist(os.environ['MLLM_EPD_EC_PATH'],key,consumer_plan)
        prompt=dict(inputs[0]);kwargs=prompt.pop('mm_kwargs');placeholders=prompt.pop('mm_placeholders')
        assert set(kwargs)==set(placeholders)==set(prompt['mm_hashes'])=={'image'}, 'Only original image-only Qwen2.5-VL inputs are supported'
        assert all(item is not None for item in kwargs['image']), 'A producer-side omitted processor item is unsupported'
        if mode=='grid_only':
            assert all(set(item)=={'pixel_values','image_grid_thw'} for item in kwargs['image']), 'Unknown Qwen input field. Do not silently omit it.'
            kwargs=MultiModalKwargsItems(image=[MultiModalKwargsItem(image_grid_thw=item['image_grid_thw'])
                                               for item in kwargs['image']])
        payload=RenderBundle(input_fingerprint(request),mode,conversation,prompt,kwargs,
            {key:list(value) for key,value in placeholders.items()})
        folder=self.directory(key);folder.mkdir(parents=True,mode=0o700,exist_ok=False)
        buffers=MsgpackEncoder().encode(payload);record=[]
        for i,buffer in enumerate(buffers):
            view=memoryview(buffer)
            target=folder/f'buffer_{i}.bin'
            with target.open('xb') as stream:stream.write(view)
            record.append(dict(name=target.name,bytes=view.nbytes,SHA256=hashlib.sha256(view).hexdigest()))
        manifest=dict(key=key,fingerprint=payload.fingerprint,mode=mode,buffers=record,
            prompt_tokens=len(prompt['prompt_token_ids']),images=len(kwargs['image']),
            image_hashes=list(prompt['mm_hashes']['image']),
            input_keys=sorted(inputs[0]),retained_mm_fields=[sorted(item) for item in kwargs['image']],
            scope='Online E-produced original token and image-grid metadata. Full E input is not modified. '
                'Grid-only consumers still require all original E embeddings through Strict EC.')
        if consumer_plan is not None:
            manifest['token_split_plan']={k:consumer_plan[k] for k in ['prefix_tokens','full_prompt_tokens']}
            manifest['image_consumer_plan']=consumer_plan['images']
        write(folder/'manifest.json',manifest)
        return manifest

    def load(self,key,request,arrival_time):
        folder=self.directory(key);manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['key']==key and manifest['fingerprint']==input_fingerprint(request), 'Handoff belongs to another input contract'
        if 'token_split_plan' in manifest:
            from .token_split import PLAN_KEY
            assert (request.kv_transfer_params or {}).get(PLAN_KEY)==manifest['token_split_plan'], 'Consumer query ownership differs from E delivery plan'
        buffers=[]
        for item in manifest['buffers']:
            assert re.fullmatch('buffer_[0-9]+[.]bin',item['name'])
            path=folder/item['name']
            assert not path.is_symlink() and path.stat().st_uid==os.getuid()
            data=path.read_bytes()
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['SHA256'], 'Corrupt processed prompt buffer'
            buffers.append(bytearray(data))
        decoder=MsgpackDecoder(RenderBundle,share_mem=False)
        decoder.pin_tensors=False
        payload=decoder.decode(buffers)
        assert payload.fingerprint==manifest['fingerprint'] and payload.mode==manifest['mode']
        prompt=dict(payload.prompt,mm_kwargs=payload.mm_kwargs,mm_placeholders=payload.mm_placeholders)
        assert set(prompt)==set(manifest['input_keys'])
        prompt['arrival_time']=arrival_time
        return payload.conversation,[prompt]


def verify_online_exports(inputs,ec_path):
    for prompt in inputs:
        for items in prompt['mm_hashes'].values():
            for identifier in items:
                assert isinstance(identifier,str) and '/' not in identifier and identifier not in ('.','..')
                folder=Path(ec_path)/identifier
                path=folder/('cuda_ipc_ready.json' if os.environ.get('MLLM_EPD_EC_TRANSPORT')=='cuda_ipc'
                    else 'encoder_cache.safetensors')
                assert path.is_file(), 'Original online E embedding export is missing: '+identifier
                if os.environ.get('MLLM_EPD_EC_TRANSPORT')=='cuda_ipc':
                    assert not (folder/'cuda_ipc_released.json').exists(),'Original E export was already released'


def install():
    from vllm.renderers.online_renderer import OnlineRenderer
    original=OnlineRenderer.render_chat
    role=os.environ['MLLM_EPD_ROLE']
    root=os.environ['MLLM_EPD_METADATA_PATH']
    store=RenderStore(root)
    ec_path=os.environ['MLLM_EPD_EC_PATH']
    async def render(renderer,request,*args,**kwargs):
        context=HANDOFF_CONTEXT.get()
        if context is None:return await original(renderer,request,*args,**kwargs)
        key,mode=context
        begin=time.monotonic();arrival=time.time()
        assert role in ('encoder','prefill','decode','pd')
        if role=='encoder':
            result=await original(renderer,request,*args,**kwargs)
            manifest=await asyncio.to_thread(store.save,key,request,result,mode)
            action='PRODUCE_ORIGINAL_PROCESSOR_RESULT'
        else:
            result=await asyncio.to_thread(store.load,key,request,arrival)
            verify_online_exports(result[1],ec_path)
            manifest=json.loads((store.directory(key)/'manifest.json').read_text())
            assert manifest['mode']==mode
            action='CONSUME_SAME_REQUEST_PROCESSOR_RESULT'
        append(ROOT/'raw'/f'processor_handoff_{role}_gpu{os.environ["CUDA_VISIBLE_DEVICES"]}.jsonl',
            dict(key=key,role=role,mode=mode,action=action,begin=begin,end=time.monotonic(),
                prompt_tokens=manifest['prompt_tokens'],images=manifest['images'],
                retained_mm_fields=manifest['retained_mm_fields'],
                buffers_bytes=sum(b['bytes'] for b in manifest['buffers']),
                manifest_SHA256=digest(store.directory(key)/'manifest.json')))
        return result
    OnlineRenderer.render_chat=render
    return original
