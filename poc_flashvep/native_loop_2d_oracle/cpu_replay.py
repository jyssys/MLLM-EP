"""Test the real native progress seam without a CUDA device."""
import sys
from types import SimpleNamespace as NS
from .common import *


def main():
    initialize()
    sys.path.insert(0, str(parent.prior.RLT))
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import Request, RequestPhase
    rows = []
    for model, depth, cut in (('ouro', 4, 3), ('huginn', 32, 31), ('ouro', 4, 2), ('huginn', 32, 28)):
        guard = None
        try:
            Scheduler._apply_wavefront_prefill_batch_progress(NS(),
                RequestPhase.PREFILLING_RECURRENT, [NS(token_start=0, request=NS())], loop_count=cut)
        except ValueError as error:
            guard = str(error)
        assert guard == 'wavefront prefill requires one recurrent depth per scheduling step'
        r = NS(request_id='r', prompt_token_ids=list(range(32)), num_prefilled_tokens=0,
               num_preluded_tokens=0, prefill_depth_ranges=Request.prefill_boundaries(32, depth),
               prefill_reclaimable_tokens=0, loops_done=0)
        queue, published = [], []
        scheduler = NS(total_ut_steps=depth, enqueue=lambda request, phase: queue.append(phase),
            cache_manager=NS(publish_prefix=lambda *a: published.append(a)),
            _prompt_destination=lambda request: RequestPhase.TRANSFERRING)
        item = NS(request=r, token_start=0, token_count=32, depth=0)
        Scheduler._apply_split_prefill_progress(scheduler, r, RequestPhase.PREFILLING_PRELUDE,
                                              item, loop_count=cut)
        assert r.num_prefilled_tokens == 0 and r.num_preluded_tokens == 32 and not published
        assert all(r.prefill_depth_ranges[i][0] == 32 for i in range(cut))
        item.depth = cut
        Scheduler._apply_split_prefill_progress(scheduler, r, RequestPhase.PREFILLING_RECURRENT,
                                              item, loop_count=depth-cut)
        assert r.num_prefilled_tokens == 32 and r.loops_done == depth and len(published) == 1
        rows.append(dict(model=model, depth=depth, cut=cut, original_wavefront_guard=guard,
                         nonwavefront_grouped_frontier=True))
    write(ROOT / 'state/native_progress_cpu_replay.json', dict(passed=True, cases=rows,
        scope='Actual native progress functions. No GPU arithmetic, transfer, output, or performance qualification.'))
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
