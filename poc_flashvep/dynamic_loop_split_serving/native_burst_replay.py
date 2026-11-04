"""Compare real native frontier advancement for grouped and single passes."""
import sys
from types import SimpleNamespace as Namespace
from .common import *


def main():
    initialize()
    sys.path.insert(0, str(prior.RLT))
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import RequestPhase
    from .loop_burst import install
    def fixture():
        request = Namespace(request_id='cpu_replay', prompt_token_ids=range(2048),
            prefill_window_origin=0, num_preluded_tokens=2048, num_prefilled_tokens=0,
            prefill_depth_ranges=[[0, 2048]] + [[0, 0] for _ in range(31)], arc_export_cut=31)
        return Namespace(request=request, phase=RequestPhase.PREFILLING_RECURRENT,
                         token_start=0, token_count=2048, depth=0)
    owner = Namespace(config=Namespace(mode='refill'), wavefront_chunked_prefill=True,
                      total_ut_steps=32, enqueue=lambda request, phase: None)
    legacy, grouped = fixture(), fixture()
    before = Scheduler._planned_loop_count(owner, [legacy])
    assert before == 1
    for depth in range(31):
        legacy.depth = depth
        Scheduler._apply_split_prefill_progress(owner, legacy.request, legacy.phase,
                                               legacy, loop_count=1)
    install(dict(contiguous_prompt_loop_burst=True, phase='CPU_FRONTIER_DIAGNOSTIC'))
    after = Scheduler._planned_loop_count(owner, [grouped])
    assert after == 31
    Scheduler._apply_split_prefill_progress(owner, grouped.request, grouped.phase,
                                           grouped, loop_count=after)
    assert legacy.request.prefill_depth_ranges == grouped.request.prefill_depth_ranges
    assert grouped.request.prefill_depth_ranges[30][0] == 2048
    assert grouped.request.prefill_depth_ranges[31] == [0, 2048]
    result = dict(passed=True, original_planned_loop_count=before,
        candidate_planned_loop_count=after, original_submissions=31, candidate_submissions=1,
        exact_native_frontier_progress=True, recurrent_passes=31,
        scope='CPU replay of native scheduler methods. No GPU arithmetic, KV byte identity, or latency claim.',
        candidate_SHA256=digest(SOURCE / 'loop_burst.py'))
    write(ROOT / 'state/native_burst_cpu_replay.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('native_grouped_loop_frontier_replay'):
        main()
