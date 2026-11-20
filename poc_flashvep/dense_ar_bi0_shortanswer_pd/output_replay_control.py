"""Keep equal-output-work controls separate from Natural EOS serving."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .axis_timeline import analyze


def main():
    dataset=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    reference=json.loads((ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json').read_text())['rows']
    length_map={r['prompt_id']:r['output_tokens'] for r in reference}
    lengths=[length_map[r['prompt_id']] for r in dataset]
    design=ROOT/'configs/frozen_equal_output_work_control_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),N=200,QPS=2.35,arrival_seed=2026112062,
            output_lengths=lengths,total_output_tokens=sum(lengths),ignore_eos=True,
            choices='Discovery-frozen best tested cut independently per axis. Same source-credit4 and actual batches1/1/1.',
            scope='Diagnostic only. Fixed lengths come from the already-frozen C1 Natural reference. '
            'No primary Natural output is replaced. Forced continuation past EOS is not a task-quality benchmark. '
            'One independent start per arm does not establish a performance CI.'))
    while not (ROOT/'state/paired_shared_KV128_confirmation_v2_complete.json').exists():
        for marker in ('STOP_REQUESTED.json','confirmation_hold.json','controlled_discovery_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    choices=json.loads((ROOT/'configs/frozen_static_axis_selection_v2.json').read_text())['winners']
    arms=[('Native',0)]+[(axis,r['cut']) for axis,r in choices.items() if 'cut' in r]
    results=[]
    for axis,cut in arms:
        name=f'EqualOutputControl_{axis}{cut}_N200_q2.35_s2026112062_BI0_v1'
        extra=dict(axis_kind=axis,axis_cut=cut,axis_state_capture=False,hidden_source_slots=4,
            qualified_controlled_performance=True,runtime_qualification_tag='fast128_v3',
            token_teacher_gate_tag='fast128_v3',burst_gate_tag='fast128_v3',qualified_async_runtime=True,
            axis_mode='STATIC') if cut else {}
        value=serving(name,axis,'longbench_shortanswer200_v2.json',200,2.35,2026112062,
            controlled_pipeline=True,source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',
            async_scheduling=True,KV_block_tokens=128,restore_official_UCX_init=True,
            diagnostic_controlled_output_replay=True,replay_output_lengths=lengths,
            phase='DIAGNOSTIC_EQUAL_OUTPUT_WORK_CONTROL',**extra)
        timeline=analyze(name);assert timeline['all_prompt_work_accounted']
        raw=[json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
        assert all(r['output_tokens']==lengths[r['sample_id']] for r in raw)
        results.append(dict(name=name,axis=axis,cut=cut,metrics=value))
        write(ROOT/'analysis/BI0_equal_output_work_control_v1.json',dict(results=results,
            evidence='DIAGNOSTIC',scope='Exactly equal output-token counts, one start per arm. Not Natural EOS quality or an independent-start confidence result.'))
    write(ROOT/'state/equal_output_control_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/equal_output_control_hold.json',dict(time=time.time(),error=repr(error)));raise
