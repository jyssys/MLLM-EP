"""Use native multi-loop submissions for one complete legal prompt region.

This is a shared scheduler optimization, not a Loop-specific advantage.
Prelude remains a separate operation. Native hidden storage and Coda guards
therefore remain on their qualified staged execution paths.
"""


def remaining_burst(item, full_depth):
    request = item.request
    start, end = item.token_start, item.token_start + item.token_count
    target_end = min(len(request.prompt_token_ids),
                     getattr(request, 'ltp_prefix', len(request.prompt_token_ids)))
    target_depth = getattr(request, 'arc_export_cut', full_depth)
    depth = item.depth
    if (start != request.prefill_window_origin or end != target_end
            or not isinstance(depth, int) or not 0 <= depth < target_depth <= full_depth):
        return None
    boundaries = request.prefill_depth_ranges
    if request.num_preluded_tokens < end or boundaries[depth][1] < end:
        return None
    if any(boundaries[index][0] != start for index in range(depth, target_depth)):
        return None
    return target_depth - depth


def install(cfg):
    if not cfg.get('contiguous_prompt_loop_burst'):
        return
    if cfg.get('phase') not in ('NATIVE_MULTI_LOOP_SUBMISSION_N8_EXACTNESS_NOT_LATENCY',
                                'CPU_FRONTIER_DIAGNOSTIC'):
        raise RuntimeError('Grouped prompt depths failed the native wavefront progress guard. Diagnostic use only.')
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import RequestPhase
    if getattr(Scheduler, '_serving_loop_burst_installed', False):
        return
    Scheduler._serving_loop_burst_installed = True
    original = Scheduler._planned_loop_count

    def planned(self, scheduled):
        phases = {item.phase for item in scheduled}
        if (self.config.mode == 'refill' and self.wavefront_chunked_prefill
                and phases == {RequestPhase.PREFILLING_RECURRENT}):
            counts = [remaining_burst(item, self.total_ut_steps) for item in scheduled]
            if counts and all(count is not None for count in counts):
                return min(counts)
        return original(self, scheduled)

    Scheduler._planned_loop_count = planned
