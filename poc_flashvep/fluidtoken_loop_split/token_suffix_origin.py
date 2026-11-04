"""CPU-tested candidate for imported suffix chunk origins. GPU-unqualified."""


def make_task(maximum, phase_type, task_type):
    def task(scheduler, request, phase, token_budget, *, start, depth=None):
        length = len(request.prompt_token_ids)
        if length > maximum:
            raise ValueError('Prompt exceeds the retained-state capacity')
        origin = request.prefill_window_origin
        if start < origin:
            raise RuntimeError('Prefill work precedes the imported-prefix origin')
        chunk = scheduler.config.prefill_chunk_size
        chunk_start = origin + (start - origin) // chunk * chunk
        end = min(length, chunk_start + chunk)
        if phase == phase_type.PREFILLING_RECURRENT:
            if depth is None:
                raise ValueError('A recurrent task needs its depth')
            if depth >= getattr(request, 'arc_export_cut', scheduler.total_ut_steps):
                return None
            left, right = request.prefill_depth_ranges[depth]
            if left != start:
                raise RuntimeError('A recurrent range has a stale frontier')
            end = min(end, right, request.num_preluded_tokens)
        count = min(token_budget, end - start)
        if count <= 0:
            return None
        return task_type(request, start, count, depth=depth, chunk_start=chunk_start,
                         prefill_hidden_row_offset=start,
                         completes_prompt=(phase == phase_type.PREFILLING_RECURRENT
                             and depth == scheduler.total_ut_steps - 1 and start + count == length))
    return task


def install(cfg):
    # No recorded GPU run enables this new candidate.
    if not cfg.get('token_suffix_origin_candidate', False):
        return
    if not cfg.get('split_evaluation') or cfg.get('axis') != 'token':
        raise ValueError('The candidate applies only to explicit Token Split tests')
    from vllm_rlt.core.scheduler import Scheduler, ScheduledRequest
    from vllm_rlt.request import RequestPhase
    previous = Scheduler._make_wavefront_task
    fixed = make_task(cfg['state_capacity_tokens'], RequestPhase, ScheduledRequest)

    def task(scheduler, request, phase, token_budget, *, start, depth=None):
        if request.prefill_window_origin == 0:
            return previous(scheduler, request, phase, token_budget, start=start, depth=depth)
        return fixed(scheduler, request, phase, token_budget, start=start, depth=depth)
    Scheduler._make_wavefront_task = task
