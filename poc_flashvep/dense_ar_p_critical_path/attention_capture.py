"""Capture real paged-FlashAttention inputs without changing model outputs."""
import argparse
import json
import os
from pathlib import Path
import re
import time
from .common import ROOT,MODEL,MODEL_REVISION,write,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());folder=ROOT/'attention_poc/raw'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(1)
    from vllm import LLM,SamplingParams
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    data=json.loads((ROOT/'data/longbench_balanced400_v2.json').read_text());byid={row['prompt_id']:row for row in data}
    assert digest(ROOT/'data/longbench_balanced400_v2.json')==cfg['dataset_SHA256']
    active={};captures=[];original=FlashAttentionImpl.forward;llm=None
    def observed(self,layer,query,key,value,kv_cache,attn_metadata,output,*a,**kw):
        match=re.search(r'model\.layers\.(\d+)\.',layer.layer_name)
        chosen=bool(active) and match and int(match[1]) in cfg['layers'] and attn_metadata is not None
        record=None
        if chosen:
            n=int(attn_metadata.num_actual_tokens);length=int(attn_metadata.seq_lens[0].item())
            if n>1 and length<=active['input_tokens']:
                assert attn_metadata.query_start_loc.numel()==2,'Capture only actual request Batch1'
                assert query.shape[1:]==(32,128) and kv_cache.shape[1]==8 and kv_cache.shape[2]==128
                block=128;blocks=(length+block-1)//block
                ids=attn_metadata.block_table[0,:blocks].long()
                packed=kv_cache.index_select(0,ids).detach().cpu()
                record=dict(prompt_id=active['prompt_id'],input_tokens=active['input_tokens'],layer=int(match[1]),
                    query_start=length-n,query_tokens=n,key_tokens=length,block_tokens=block,
                    scale=float(self.scale),causal=bool(attn_metadata.causal),fa_version=int(self.vllm_flash_attn_version),
                    num_splits=int(attn_metadata.max_num_splits),aot_schedule=attn_metadata.scheduler_metadata is not None,
                    sliding_window=list(self.sliding_window) if self.sliding_window is not None else None,
                    softcap=float(self.logits_soft_cap),original_query_strides=list(query.stride()),
                    original_KV_strides=list(kv_cache.stride()),Q=query[:n].detach().cpu(),packed_KV=packed)
                assert self.alibi_slopes is None and getattr(self,'sinks',None) is None
                assert record['causal'] and record['fa_version']==3
                assert record['sliding_window'] in (None,[-1,-1]),'Only exact global causal attention is supported'
        result=original(self,layer,query,key,value,kv_cache,attn_metadata,output,*a,**kw)
        if record is not None:
            record['output']=output[:record['query_tokens']].detach().cpu()
            name=f'capture_{len(captures):04d}.pt';path=folder/name;torch.save(record,path)
            captures.append(dict(path=str(path.relative_to(ROOT/'attention_poc')),SHA256=digest(path),
                **{key:value for key,value in record.items() if key not in ('Q','packed_KV','output')}))
        return result
    FlashAttentionImpl.forward=observed
    options=dict(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=40960,max_num_batched_tokens=8192,max_num_seqs=1024,kv_cache_memory_bytes=42*1024**3,
        enable_prefix_caching=False,enable_chunked_prefill=True,async_scheduling=True,
        seed=2026112450,disable_log_stats=True,enforce_eager=True,attention_config=dict(backend='FLASH_ATTN'),block_size=128)
    outputs=[]
    try:
        write(folder/'status.json',dict(status='STARTING',pid=os.getpid(),time=time.time()))
        llm=LLM(**options)
        actual=llm.llm_engine.vllm_config
        write(folder/'actual_runtime.json',dict(options=options,model_revision=MODEL_REVISION,
            B=actual.scheduler_config.max_num_batched_tokens,S=actual.scheduler_config.max_num_seqs,
            device=torch.cuda.get_device_name(0),UUID=str(torch.cuda.get_device_properties(0).uuid),
            source_SHA256=digest(Path(__file__)),scope='Original full36 model. Eager capture is correctness-only,not a compiled-C3 timing baseline.'))
        for selected in cfg['prompts']:
            row=byid[selected['prompt_id']];assert row['input_tokens']==selected['input_tokens']
            active.update(row)
            answer=llm.generate([dict(prompt_token_ids=row['input_ids']['qwen8'])],
                SamplingParams(temperature=0,max_tokens=16,ignore_eos=True,seed=2026112450),use_tqdm=False)[0]
            active.clear();outputs.append(dict(prompt_id=row['prompt_id'],output_ids=list(answer.outputs[0].token_ids)))
        for selected in cfg['prompts']:
            for layer in cfg['layers']:
                rows=sorted((row for row in captures if row['prompt_id']==selected['prompt_id'] and row['layer']==layer),key=lambda row:row['query_start'])
                cursor=0
                for row in rows:assert row['query_start']==cursor;cursor+=row['query_tokens']
                assert cursor==selected['input_tokens'],(selected['prompt_id'],layer,cursor)
        write(folder/'capture_manifest.json',dict(passed=True,captures=captures,outputs=outputs,model_revision=MODEL_REVISION,
            dataset_SHA256=cfg['dataset_SHA256'],scope='Actual post-Q/K-normalization and RoPE queries,valid paged KV and original attention outputs. '
                'No padding,filler or prompt modification. Keep real B8192 chunk boundaries. Gather valid KV blocks,including unused final-block storage. '
                'Only key_tokens positions are valid. Capture CPU copies are intrusive and have no timing claim. '
                'All36 native model layers still execute. No remote result is inserted into this capture model.'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time(),captures=len(captures)))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error)));raise
    finally:
        FlashAttentionImpl.forward=original
        if llm is not None:llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
