"""Measure real BI0 layer APIs with valid causal KV in one stock engine.

This diagnostic excludes transport, queues, LM head and sampling. It does not
implement online offloading. Prefix caching only constructs the token-prefix KV.
"""
import json
import os
import sys
import time
from .common import ROOT, MODEL_REVISION, write
from .engine_options import options


def main():
    cfg=json.load(open(sys.argv[1]));folder=ROOT/'raw'/cfg['name']
    assert os.environ['VLLM_BATCH_INVARIANT']=='0'
    import torch
    torch.set_num_threads(1)
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.qwen3 import Qwen3Model
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from .activity import Activity
    state=dict(enabled=False);rows=[];primitive_rows=[];engine=None;collector=None
    original_prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*args,**kwargs):
        value=original_prepare(self,*args,**kwargs);state['batch']=value
        return value
    GPUModelRunner.prepare_inputs=prepared
    original_forward=Qwen3Model.forward
    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        if not state['enabled']:
            return original_forward(self,input_ids,positions,intermediate_tensors,inputs_embeds)
        state['enabled']=False
        batch=state['batch'];m=state['cut'];length=state['length'];axis=state['axis']
        start=int(batch.num_computed_tokens_np[0]);queries=int(batch.num_scheduled_tokens[0])
        assert int(batch.num_reqs)==1
        assert queries==(length if axis=='Layer' else length*m//36)
        assert start==(0 if axis=='Layer' else length-queries)
        assert int(positions[0])==start and int(positions[queries-1])==length-1
        metadata=dict(axis=axis,cut=m,input_tokens=length,valid_queries=queries,
            actual_queries=int(input_ids.shape[0]),padded_queries=int(batch.num_tokens_after_padding),
            query_start=start,actual_batch=1,valid_GEMM_rows=length*m,
            observed_token_GEMM_rows=36*queries if axis=='Token' else None)
        # The ordinary full forward defines the output. Replays cannot replace it.
        if axis=='Token':
            baseline=original_forward(self,input_ids,positions,intermediate_tensors,inputs_embeds)
            def replay():
                return original_forward(self,input_ids,positions,intermediate_tensors,inputs_embeds)
            restore=lambda:None
        else:
            assert intermediate_tensors is None and inputs_embeds is None
            hidden=self.embed_input_ids(input_ids);residual=None;boundary=None
            for index in range(36):
                if index==36-m:
                    boundary=(hidden.detach().clone(),residual.detach().clone())
                hidden,residual=self.layers[index](positions,hidden,residual)
            baseline,_=self.norm(hidden,residual)
            assert boundary is not None
            work=(torch.empty_like(boundary[0]),torch.empty_like(boundary[1]))
            def restore():
                work[0].copy_(boundary[0]);work[1].copy_(boundary[1])
            def replay():
                h,r=work
                for index in range(36-m,36):h,r=self.layers[index](positions,h,r)
                h,_=self.norm(h,r)
                return h
        # GPU copies that restore owned input buffers are outside the service ROI.
        # RMSNorm can modify its input. Every replay therefore restores the boundary.
        def pass_cost(label,activity=None):
            samples=[];equal=True
            for repeat in range(cfg.get('repeats',12)):
                restore();torch.cuda.synchronize()
                left=activity.stamp() if activity else None
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();value=replay();end.record();end.synchronize()
                right=activity.stamp() if activity else None
                equal=equal and bool(torch.equal(value[:queries],baseline[:queries]))
                samples.append(dict(CUDA_enclosure_ms=begin.elapsed_time(end),
                    CUPTI_left=left,CUPTI_right=right,warmup=repeat<2))
            rows.append(dict(metadata,pass_label=label,samples=samples,strict_hidden_equal=equal,
                scope='Actual vLLM BI0 decoder-layer APIs. CUDA events include launch gaps. '
                'No transport, queue, LM-head, sample or online serving speedup.'))
            assert equal,'An isolated replay changed the same-geometry hidden tensor'
        pass_cost('clean_before')
        if cfg.get('cost_kernel_activity',True):
            assert collector is not None
            assert collector.lib.activity_enable(0)==0
            try:pass_cost('profile',collector)
            finally:collector.pause()
        pass_cost('clean_after')
        if collector is not None:
            # Identify kernels through the actual serving Linear API. These
            # diagnostic calls do not contribute to the forward-cost samples.
            layer=self.layers[-1]
            for component,module in (
                ('QKV',layer.self_attn.qkv_proj),('O',layer.self_attn.o_proj),
                ('MLP_gate_up',layer.mlp.gate_up_proj),('MLP_down',layer.mlp.down_proj)):
                weight=module.weight
                x=torch.zeros((int(input_ids.shape[0]),int(weight.shape[1])),device='cuda',dtype=weight.dtype)
                torch.cuda.synchronize();assert collector.lib.activity_enable(0)==0
                left=collector.stamp();out=module(x);torch.cuda.synchronize();right=collector.stamp()
                collector.pause()
                primitive_rows.append(dict(metadata,component=component,
                    Linear_class=type(module).__name__,quant_method=type(module.quant_method).__name__,
                    input_shape=list(x.shape),weight_shape=list(weight.shape),
                    CUPTI_left=left,CUPTI_right=right,
                    scope='Kernel identity through the serving Linear API. Zero input is diagnostic. Not a serving cost or correctness result.'))
        write(folder/'isolated_axis_costs_partial.json',dict(rows=rows))
        return baseline
    Qwen3Model.forward=forward
    try:
        write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
        opts=options(dict(cfg,execution_family='MATCHED_NO_COMPILE'))
        opts.update(enable_prefix_caching=True,async_scheduling=False,
            compilation_config=dict(mode=0,cudagraph_mode='NONE'))
        llm=LLM(**opts);engine=llm.llm_engine
        if cfg.get('cost_kernel_activity',True):
            sub=folder/'activity_all';sub.mkdir()
            collector=Activity(sub,include_copies=False);collector.pause()
        actual=engine.vllm_config
        write(folder/'actual_runtime.json',dict(options=opts,model_revision=MODEL_REVISION,
            actual_B=actual.scheduler_config.max_num_batched_tokens,
            actual_S=actual.scheduler_config.max_num_seqs,batch_invariant=False,
            GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
            scope='Isolated eager model service. Prefix caching constructs valid causal KV only. Not a Main baseline.'))
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
        source=max(data,key=lambda r:r['input_tokens'])
        params=SamplingParams(max_tokens=1,ignore_eos=True,temperature=0,detokenize=False)
        def generate(ids):
            return llm.generate([dict(prompt_token_ids=ids)],params,use_tqdm=False)
        for length in cfg.get('lengths',[4608,8064]):
            assert length%576==0 and length<=8192
            ids=source['input_ids']['qwen8'][:length]
            assert len(ids)==length
            for cut in cfg.get('cuts',[1,2,4,6,8,12]):
                suffix=length*cut//36;prefix=length-suffix
                assert prefix%16==0 and suffix%16==0 and 36*suffix==length*cut
                for axis in ('Layer','Token'):
                    assert llm.reset_prefix_cache()
                    if axis=='Token':generate(ids[:prefix])
                    state.update(enabled=True,axis=axis,cut=cut,length=length)
                    generate(ids)
                    assert not state['enabled']
                    print('COST',length,cut,axis,flush=True)
        write(folder/'isolated_axis_costs.json',dict(rows=rows,primitive_identity=primitive_rows,source_prompt_id=source['prompt_id'],
            source_dataset=cfg['dataset'],logical_rows_exact=True,actual_batch=1,
            evidence='ISOLATED_FORWARD_MEASUREMENT',
            scope='Real prompt-token prefixes. Full36 token suffix has valid cached-prefix KV. '
            'Last-m uses the full-query layer boundary. In-process repetitions are not independent server starts.'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error)));raise
    finally:
        if collector is not None:collector.close()
        if engine is not None:engine.engine_core.shutdown()


if __name__=='__main__':main()
