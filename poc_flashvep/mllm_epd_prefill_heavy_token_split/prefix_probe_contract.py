"""Real prefix compute for isolated cost and interference probes, not answers."""
import os
import time

KEY='campaign_isolated_prefix_probe_v1'


def truncate_probe(request,role):
    params=(request.sampling_params.extra_args or {}).get('kv_transfer_params') or {}
    plan=params.get(KEY)
    if plan is None:return request
    assert role=='pd' and set(plan)=={'prefix_tokens','full_prompt_tokens'}
    prefix,full=plan['prefix_tokens'],plan['full_prompt_tokens']
    assert type(prefix) is int and type(full) is int and 0<prefix<full
    assert len(request.prompt_token_ids)==full
    assert request.sampling_params.max_tokens==1,'The single prefix sample is a discarded probe, not a task answer'
    assert not params.get('do_remote_prefill') and not params.get('do_remote_decode')
    assert 'campaign_token_split_v1' not in params and 'campaign_boundary_teacher_v1' not in params
    request.prompt_token_ids=request.prompt_token_ids[:prefix]
    return request


def install():
    from vllm.v1.engine.input_processor import InputProcessor
    from .common import ROOT,append
    role=os.environ['MLLM_EPD_ROLE'];assert role=='pd'
    original=InputProcessor.process_inputs
    def processed(processor,*args,**kwargs):
        request=original(processor,*args,**kwargs)
        before=request.prompt_token_ids
        result=truncate_probe(request,role)
        if result.prompt_token_ids is not before:
            append(ROOT/'raw'/f'isolated_prefix_probe_{os.getpid()}.jsonl',dict(time=time.monotonic(),
                request_id=result.request_id,full_prompt_tokens=len(before),prefix_tokens=len(result.prompt_token_ids),
                original_MM_features_retained=len(result.mm_features or []),
                evidence='ISOLATED_PREFILL_PROBE',task_answer='NOT_GENERATED',
                scope='Original full-input validation precedes truncation. All original MM features remain for MRoPE. '
                    'The prefix KV is real, but no original request is completed by this discarded sample.'))
        return result
    InputProcessor.process_inputs=processed
    return original
