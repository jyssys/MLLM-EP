"""Exact token ownership. Keep the original model and multimodal metadata."""
import hashlib
import json
import os
import time


PLAN_KEY = 'campaign_token_split_v1'
POLICY_KEY = 'campaign_token_split_policy_v1'
TEACHER_KEY = 'campaign_boundary_teacher_v1'


def select_prefix(length, suffix_fraction=0., fixed_prefix=0):
    assert type(length) is int and length > 1
    assert 0 <= suffix_fraction < 1 and type(fixed_prefix) is int and fixed_prefix >= 0
    assert not (suffix_fraction and fixed_prefix), 'Use a ratio or a correctness boundary, not both'
    if fixed_prefix:
        assert fixed_prefix < length
        return fixed_prefix
    return max(1, min(length - 1, int(round(length * (1 - suffix_fraction))))) if suffix_fraction else None


def split_plan(params):
    plan = (params or {}).get(PLAN_KEY)
    if plan is None:
        return None
    assert set(plan) == {'prefix_tokens', 'full_prompt_tokens'}, 'Unknown token ownership contract'
    prefix, full = plan['prefix_tokens'], plan['full_prompt_tokens']
    assert type(prefix) is int and type(full) is int and 0 < prefix < full
    return prefix, full


def truncate_producer_request(request, role):
    """Shorten only P query ownership after original full-input validation."""
    sampling = request.sampling_params
    params = (sampling.extra_args or {}).get('kv_transfer_params') if sampling is not None else None
    plan = split_plan(params)
    if plan is None:
        return request
    prefix, full = plan
    assert request.prompt_token_ids is not None and len(request.prompt_token_ids) == full
    if role != 'prefill':
        assert role == 'decode', 'Only P and D may consume a token ownership plan'
        return request
    assert params.get('do_remote_decode') is True and not params.get('do_remote_prefill')
    assert sampling.max_tokens == 1, 'Discard the single P sample. D performs the first actual sampling.'
    # Full original features retain image grids, spans and order for MRoPE.
    # The original scheduler loads only images that overlap actual query rows.
    request.prompt_token_ids = request.prompt_token_ids[:prefix]
    return request


def matched_remote_prefix(request, num_computed_tokens):
    """Return None for Native. A marked D request imports exactly P's prefix."""
    params = request.kv_transfer_params
    plan = split_plan(params)
    if plan is None or not params.get('do_remote_prefill'):
        return None
    prefix, full = plan
    assert len(request.prompt_token_ids or []) == request.num_prompt_tokens == full
    assert type(params.get('remote_num_tokens')) is int and params['remote_num_tokens'] == prefix, (
        'The producer did not compute the advertised exact prefix', params.get('remote_num_tokens'), prefix)
    assert params.get('remote_block_ids'), 'A real NIXL prefix handoff is required'
    assert type(num_computed_tokens) is int and num_computed_tokens >= 0
    count = max(0, prefix - num_computed_tokens)
    return count, count > 0


def teacher_query_limit(request, computed, count):
    """Align an unsplit correctness teacher's chunks. Never use for timing."""
    params = request.kv_transfer_params or {}
    teacher = params.get(TEACHER_KEY)
    if teacher is None:
        return count
    prefix, full = split_plan({PLAN_KEY: teacher})
    assert PLAN_KEY not in params and not params.get('do_remote_prefill') and not params.get('do_remote_decode')
    assert len(request.prompt_token_ids or []) == request.num_prompt_tokens == full
    assert computed >= 0 and count >= 0
    return min(count, prefix - computed) if computed < prefix else count


def install_boundary_teacher():
    from vllm.v1.core.sched.scheduler import Scheduler
    original = Scheduler._try_schedule_encoder_inputs
    if getattr(original, '_campaign_boundary_teacher', False):
        return original

    def limited(scheduler, request, num_computed_tokens, num_new_tokens, *args, **kwargs):
        return original(scheduler, request, num_computed_tokens,
            teacher_query_limit(request, num_computed_tokens, num_new_tokens), *args, **kwargs)

    limited._campaign_boundary_teacher = True
    Scheduler._try_schedule_encoder_inputs = limited
    return original


def install_input_prefix():
    """API-side ownership change. Do not replace any model or attention forward."""
    from vllm.v1.engine.input_processor import InputProcessor
    from .common import ROOT, append
    original = InputProcessor.process_inputs
    role = os.environ['MLLM_EPD_ROLE']
    assert role in ('prefill', 'decode')

    def process(processor, *args, **kwargs):
        request = original(processor, *args, **kwargs)
        before = request.prompt_token_ids
        result = truncate_producer_request(request, role)
        if before is not result.prompt_token_ids:
            params = result.sampling_params.extra_args['kv_transfer_params']
            append(ROOT / 'raw' / f'token_ownership_{role}_{os.getpid()}.jsonl', dict(
                time=time.monotonic(), request_id=result.request_id, plan=params[PLAN_KEY],
                original_prompt_SHA256=hashlib.sha256(json.dumps(before).encode()).hexdigest(),
                prefix_prompt_SHA256=hashlib.sha256(json.dumps(result.prompt_token_ids).encode()).hexdigest(),
                retained_image_features=len(result.mm_features or []),
                scope='API request ownership only. Not GPU execution or correctness evidence.'))
        return result

    InputProcessor.process_inputs = process
    return original
