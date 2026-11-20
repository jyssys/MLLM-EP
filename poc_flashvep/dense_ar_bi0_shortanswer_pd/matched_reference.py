"""Compare decode boundaries at the same BI0 batch and execution settings."""
import json
import time
from .common import ROOT,save,write
from .qualification import configuration,records
from .run import execute
from .metrics import summarize


def main():
    while not (ROOT/'state/fast_native_control_state_v3_complete.json').exists():
        for marker in ('STOP_REQUESTED.json','async_qualification_state_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    name='Reference_Matched_C0_Async_KV128_C1_Calibration20_BI0_v1'
    cfg=configuration(name,'Native','cost_calibration20.json',count=20,gpus=[1],reference_single=True,
        state_capture=False,correctness_only=False,execution_family='MATCHED_NO_COMPILE',
        async_scheduling=True,KV_block_tokens=128,restore_official_UCX_init=True,
        phase='MATCHED_SINGLE_GPU_DECODE_REFERENCE')
    assert execute(save(cfg)),name
    reference=summarize(name)
    native='ContinuationCost_Native0_calibration20_BI0_v1'
    file=ROOT/'raw'/native/'dispatcher_result.json'
    while not file.exists():
        for marker in ('STOP_REQUESTED.json','continuation_costs_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    assert json.loads(file.read_text())['complete']
    measured=summarize(native)
    a=records(ROOT/'raw'/name/'requests.jsonl');b=records(ROOT/'raw'/native/'requests.jsonl')
    assert [r['prompt_id'] for r in a]==[r['prompt_id'] for r in b]
    fields=('mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens')
    write(ROOT/'analysis/BI0_matched_single_native_decode_reference_v1.json',dict(
        names=dict(single=name,native=native),independent_starts_per_arm=1,
        Single={k:reference[k] for k in fields},Native={k:measured[k] for k in fields},
        Native_over_Single={k:measured[k]/reference[k] for k in fields if reference[k]},
        strict_greedy_matches=sum(r['output_ids']==s['output_ids'] for r,s in zip(a,b)),N=20,
        evidence='DIAGNOSTIC',
        scope='Same real prompts, BI0/C0/async/full decode graph/KV128, serial actual request batch1, no state-copy profiler. '
        'Both clocks are coordinator delivery. Single uses one GPU; PD uses two. Not an equal-resource throughput comparison. '
        'Native recomputes its last prompt query. Different first-query geometry can change Natural output. '
        'No independent-start confidence interval or exclusive GPU timing is inferred.'))
    write(ROOT/'state/matched_single_native_reference_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/matched_single_native_reference_hold.json',dict(time=time.time(),error=repr(error)));raise
