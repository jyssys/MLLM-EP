"""Reproduce the actual caller guard omitted by the first CPU replay."""
import sys
from types import SimpleNamespace as Namespace
from .common import *


def main():
    initialize()
    sys.path.insert(0, str(prior.RLT))
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import RequestPhase
    rows = []
    message = 'wavefront prefill requires one recurrent depth per scheduling step'
    for model, depth, cut in (('ouro', 4, 3), ('huginn', 32, 31)):
        request = Namespace()
        item = Namespace(request=request, token_start=0)
        try:
            Scheduler._apply_wavefront_prefill_batch_progress(Namespace(),
                RequestPhase.PREFILLING_RECURRENT, [item], loop_count=cut)
        except ValueError as error:
            assert str(error) == message
            rows.append(dict(model=model, full_depth=depth, proposed_loop_count=cut,
                             rejected_by_actual_callback=True, error=str(error)))
        else:
            raise AssertionError('The unsupported native callback did not reject grouped depths')
    result = dict(passed=True, cases=rows,
        scope='Corrected CPU seam: the actual wavefront callback rejects grouped prompt depths before lifecycle advancement. The earlier single-span helper replay was insufficient. No GPU correctness or speedup claim.',
        optional_candidate_status='BLOCKED_NATIVE_WAVEFRONT_PROGRESS_GUARD')
    write(ROOT / 'state/native_burst_wavefront_guard_replay.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('actual_wavefront_grouped_depth_guard_replay'):
        main()
