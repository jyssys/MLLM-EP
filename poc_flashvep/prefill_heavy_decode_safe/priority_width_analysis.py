"""Analyze matched recurrent rows without substituting them for total FLOPs."""
import json,statistics
from .common import *


def analyze_available():
    initialize()
    from ..loop_width_gemm_validation import cost_analysis
    cost_analysis.ROOT=ROOT;cost_analysis.configure=configure
    plan=json.loads((ROOT/'state/priority_width_comparison_plan.json').read_text());costs={}
    for name in plan['jobs']:
        if '_serial8_' not in name:continue
        receipt=ROOT/'raw'/name/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        value=cost_analysis.analyze(name)
        if isinstance(value,tuple):value=value[0]
        costs[name]=value
    projection=json.loads((WIDTH/'analysis/fixed_policy_confirmation.json').read_text())
    policies={p:json.loads((WIDTH/'configs'/f'operator_kernel_policy_{p}.json').read_text()) for p in ('s1','v2')}
    matrix=[]
    for entry in plan['matrix']:
        model=entry['model'];width=entry['token_suffix'];loopcount=entry['D_loops'];L=entry['length'];R=entry['depth']
        pair={arm:costs.get(f'priority_width_{model}_S{width}_{arm}_grouped_tuned_serial8_s1') for arm in ('loop','token')}
        baseline=costs.get(f'priority_width_{model}_zero_tuned_serial8_s1');measurements={}
        for arm,value in pair.items():
            if value is None:measurements[arm]=dict(status='NOT_RUN_OR_NOT_COMPLETE');continue
            core=value['core_component_enclosures'].get('decode',{})
            per_request=value['streams']['prefill']['per_request']
            before=baseline['streams']['prefill']['per_request'] if baseline else {}
            saved=[before[s]-t for s,t in per_request.items() if s in before]
            measurements[arm]=dict(status='MEASURED' if value['passed'] else 'FAILED',passed=value['passed'],
                core_graph_sum_ms=core.get('packed_recurrent_graph'),D_prompt_sum_ms=value['streams']['decode'],
                D_prelude_ms=core.get('prelude'),D_prompt_coda_ms=core.get('prompt_coda'),
                P_saved_event_enclosure_ms=cost_analysis.distribution(saved),
                largest_D_submission_ms=value['largest_D_prompt_submission_ms'],
                D_submissions_per_request=value['D_prompt_submissions_per_request'],
                payload_bytes=value['payload_bytes_per_request'],service_TTFT_ms=value['service_TTFT'],
                handoff_host_enclosure_ms=value['handoff'],actual_D_prompt_rows=value['actual_D_prompt_rows'],
                output_parity=value['parity'],full_work=value['full_work'],lifetime=value['lifetime'],normal_worker_exit=value['normal_worker_exit'])
        old=[r for r in projection['width'] if r['model']==model and r['token_batch']==loopcount]
        def choices(policy):
            return {(r['role'],r['M'],r['N'],r['K']):(r['method'],r.get('algorithm'))
                for r in policies[policy]['models'][model]['entries'] if r['M'] in (width,L)}
        assert choices('s1')==choices('v2'),'Published dense timings use different operator choices'
        pairs_full=L*(L+1)//2*loopcount
        pairs_suffix=((L*(L+1)-(L-width)*(L-width+1))//2)*R
        row=dict(entry,measurements=measurements,structural_attention_causal_pairs_per_core_layer=dict(loop=pairs_full,token=pairs_suffix),
            fixed_policy_dense_token_over_loop_three_starts=[r['token_over_loop'] for r in old],
            relevant_S1_and_v2_operator_choices_identical=True,
            projection_scope='Reanalysis of published isolated GEMM timings at the same row geometries. Structural multiplicities match this pair. These are not fresh recurrent-core or Serving timings.',
            scope='Fresh serial CUDA Event enclosures. Same fixed tuned policy. Full remaining Loop groups are allowed on both arms. No concurrent online Decode-safety or physical-idle conclusion.')
        if all(measurements[a].get('passed') for a in ('loop','token')):
            a,b=measurements['loop'],measurements['token']
            row['Token_over_Loop_D_prompt_ratio']=b['D_prompt_sum_ms']['p50']/a['D_prompt_sum_ms']['p50']
            if a['core_graph_sum_ms'] and b['core_graph_sum_ms']:
                row['Token_over_Loop_core_ratio']=b['core_graph_sum_ms']['p50']/a['core_graph_sum_ms']['p50']
        matrix.append(row)
    write(ROOT/'analysis/priority_width_comparison_results.json',dict(matrix=matrix,
        counts=dict(completed_serial_cells=len(costs),planned_serial_cells=sum('_serial8_' in n for n in plan['jobs'])),
        scope='N8 state checks precede these diagnostics. Eight serial requests within one process are not eight independent starts. No global optimum GEMM or universal Loop superiority claim.'))
    return matrix


if __name__=='__main__':
    values=analyze_available()
    for r in values:
        print(r['model'],r['token_suffix'],{a:{k:v.get(k) for k in ('status','largest_D_submission_ms','D_submissions_per_request')} for a,v in r['measurements'].items()},
            r.get('Token_over_Loop_core_ratio'),r.get('Token_over_Loop_D_prompt_ratio'))
