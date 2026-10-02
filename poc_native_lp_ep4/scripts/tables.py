"""Generate evidence tables only from completed measured JSON artifacts."""
import argparse
from collections import defaultdict
from pathlib import Path

from poc_native_lp_ep4.contract import read_json


def number(value,digits=2):
    if value is None:return 'N/A'
    if isinstance(value,dict):
        ci=value.get('ci95')
        return f"{value['mean']:.{digits}f}"+(f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else '')
    return f'{value:.{digits}f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
        *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--stages',type=Path,help='secondary actual engine-completion stage analysis')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);d=read_json(a.metrics)
    def save(name,text):(a.output/name).write_text(text)
    final=d['final_summary'];discovery=d['discovery_summary'];best=d['selection']['best_arm']
    serve='# Measured native serving latency\n\n'
    serve+='Scope: CONTROLLED_NATIVE_LP_EP4_HARNESS. Values are mean [95% wave-bootstrap CI], ms. '
    serve+='The token clock is engine completion, never external polling. CPU rendering occurs before model-ready arrival; '
    serve+='GPU vision encoding is enabled. Model-ready TTFT and language-entry-to-first-token wall are separate. '
    serve+='Resident decode is B8, original K8, fixed128 in the final lane.\n\n'
    for title,rows in (('Frozen final (10 paired repeats)',final),('Discovery (3 repeats/configuration, fixed96)',discovery)):
        serve+='## '+title+'\n\n'
        serve+=table(['Arm','Input','n','TTFT','Mean ITL','P95 ITL','P99 ITL','Peak VSEL','Response peak ITL','LM-entry wall'],[
            [r['arm'],r['modality'],r['n'],*[number(r[k]) for k in ('ttft_ms','mean_itl_ms','p95_itl_ms','p99_itl_ms',
                'peak_vsel_ms','peak_itl_ms','lm_start_to_first_token_ms')]] for r in rows])+'\n'
    serve+='LP_AUTO follows the pinned upstream rule and is G4 for both inputs. G4 is not an independent algorithm. '
    serve+=f'Best LP was frozen as **{best}** from discovery, before final repeats. '
    serve+='The SLO proxy is peak ITL / paired decode-only median <=1.5, not a production SLO. '
    serve+='Peak statistics include the whole injection response window: finer groups have a longer exposure window. '
    serve+='These are repeated measurements of a fixed bounded pool, not claims across unseen workloads.\n'
    if a.stages:
        stages=read_json(a.stages)
        serve+='\n## First producer iteration versus later stages (secondary)\n\n'
        serve+=table(['Arm','Input','Phase','Wave n','Wave peak ITL','Paired wave peak VSEL'],[
            [r['arm'],r['modality'],r['phase'],r['n_waves'],number(r['peak_stage_itl_ms']),number(r['peak_stage_vsel_ms'])]
            for r in stages['summary']])+'\n'
        error=max(r['dp0_completion_alignment_error_ms'] for r in stages['rows'])
        serve+=f'Actual DP0 scheduler/engine completion alignment residual is at most **{error:.3f} ms**. '
        serve+='Peer ITL uses the same forced token position. This separation neither removes the first burst '
        serve+='from the primary metric nor changes the frozen winner. Later-stage peak exposure differs with group count; '
        serve+='layers/stages/residents are not independent bootstrap samples. These host clocks are not GPU CWT.\n'
    save('serving_latency.md',serve)
    ep='# Native MoE/EP component evidence\n\n'
    ep+='All numbers below are local CUDA spans. We select one cumulative-critical rank and keep dispatch/expert/combine '
    ep+='on that same rank; we do not sum unrelated rank maxima or subtract cross-device timestamps. '
    ep+='Prepare/dispatch can include peer producer-readiness waits and host layout work, not just network service. '
    ep+='The active mixed-stage kernel contains resident decode rows as well as prefill rows; fused expert time cannot be '
    ep+='allocated between them by a FLOPs ratio.\n\n'
    usable={(r['arm'],r['modality']):r['usable'] for r in d['instrumentation_overhead']['stratified']}
    ep+=table(['Arm','Input','n','Dispatch total','Expert total','Max-rank expert total','Combine total','MoE total','Max prepare span','Overhead screen'],[
        [r['arm'],r['modality'],r['n'],*[number(r[k]) for k in ('dispatch_total_ms','expert_total_ms','max_rank_expert_total_ms',
          'combine_total_ms','moe_total_ms','max_dispatch_quantum_ms')],
          'USABLE_LOCAL_SPAN' if usable.get((r['arm'],r['modality']),False) else 'UNVALIDATED'] for r in d['component_summary']])+'\n'
    ep+='## Route/weight-working-set accounting (independent rich lane)\n\n'
    ep+=table(['Arm','Input','Reload amplification','Weight-access proxy (GB)','Logical remote dispatch (GB)'],[
        [r['arm'],r['modality'],number(r['expert_reload_amp']),
         'N/A' if r['expert_weight_bytes_proxy'] is None else f"{r['expert_weight_bytes_proxy']['mean']/1e9:.3f}",
         'N/A' if r['logical_remote_dispatch_bytes'] is None else f"{r['logical_remote_dispatch_bytes']['mean']/1e9:.3f}"]
        for r in d['route_summary']])+'\n'
    ep+='Each expert is **9,437,184 bytes** from actual loaded BF16 gate/up+down tensors. '
    ep+='ReloadAmp counts reactivations of prompt experts across chunks, not actual HBM reads. '
    ep+='Payload is destination-deduplicated hidden-vector accounting; metadata/padding/wire traffic is not inferred. '
    ep+='Per-rank send/recv matrices, per-layer times, active-expert counts and weight shapes remain in raw artifacts.\n\n'
    ep+='These counters and totals attach to **prefill-active mixed kernel invocations**, not an isolated prefill-only GEMM. '
    ep+='LP also executes resident-only layers on each stage; their useful decode work is not charged to the prompt reload proxy. '
    ep+='Low reload therefore does not imply identical whole-system HBM traffic or TTFT.\n\n'
    ep+=f"Logical top8 paths across all 48 layers: {'PASS' if d['all_logical_path_invariants_pass'] else 'FAIL'}. "
    audit=d['component_frame_audit']
    ep+=f"Complete four-rank **active injected-prefill** component frames: {'PASS' if audit['all_active_prefill_frames_complete'] else 'FAIL'}. "
    ep+=f"All captured frames complete: **{audit['all_captured_frames_complete']}**; {audit['partial_frame_count']} partial frames "
    ep+=f"({audit['partial_active_prefill_frame_count']} active-prefill) are retained in the trace-frame audit, never filled. "
    ep+='Post-injection capture-boundary omissions are not inferred missing executions; native stage/token correctness is checked separately.\n\n'
    ep+='## Instrumentation overhead\n\n'
    overhead=d['instrumentation_overhead']
    ep+=table(['Metric','Trace minus hook-bypass (%)'],[[k,number(v)] for k,v in overhead['summary'].items()])+'\n'
    ep+=table(['Arm','Input','TTFT overhead (%)','Mean ITL overhead (%)','Stratum usable'],[
        [r['arm'],r['modality'],number(r['ttft_ms']),number(r['mean_itl_ms']),r['usable']]
        for r in overhead['stratified']])+'\n'
    ep+=f"Component breakdown usable under the <=10% screening rule: **{overhead['component_breakdown_usable']}**. "
    ep+='The individual failing strata are UNVALIDATED and cannot support an EP-common verdict; passing strata remain separately labeled. '
    ep+='Unhooked clean serving results above remain separate.\n'
    save('ep_component_analysis.md',ep)
    mod='# Matched-token modality comparison\n\n'
    mod+='Visual: 2029 total/1980 visual rows. Text-only: 2033 rows, same model/dtype/runtime. '
    mod+='Positive delta means visual is slower. Wave pairs are matched by arm/repeat; 48 layers are not independent samples.\n\n'
    mod+=table(['Arm','Peak VSEL delta (ms)','TTFT delta (ms)','LM-entry wall delta (ms)','Peak ITL delta (ms)'],[
        [arm,*[number(fields.get(k)) for k in ('peak_vsel_ms_delta','ttft_ms_delta','lm_start_to_first_token_ms_delta','peak_itl_ms_delta')]]
        for arm,fields in d['final_modality_deltas'].items()])+'\n'
    mod+='## Local component diagnostic\n\n'
    mod+=table(['Arm','Dispatch delta (ms)','Expert delta (ms)','Combine delta (ms)','Max dispatch span delta (ms)'],[
        [arm,*[number(fields.get(k)) for k in ('dispatch_total_ms_delta','expert_total_ms_delta','combine_total_ms_delta','max_dispatch_quantum_ms_delta')]]
        for arm,fields in d['component_modality_deltas'].items()])+'\n'
    geometry=defaultdict(list)
    for r in d['route_geometry_waves']:geometry[(r['arm'],r['modality'])].append(r)
    mod+=table(['Arm','Input','Mean destination fanout','Mean layer rank max/mean','Logical dispatch (GB, 48 layers)'],[
        [arm,mode,*[f"{sum(r[k] for r in rows)/len(rows):.3f}" for k in (
            'mean_distinct_destination_ranks_per_prompt_token','mean_layer_rank_assignment_max_mean')],
         f"{sum(r['logical_hidden_dispatch_bytes'] for r in rows)/len(rows)/1e9:.3f}"]
        for (arm,mode),rows in geometry.items()])+'\n'
    mod+='A large first-layer prepare span alone cannot identify an intrinsic visual A2A mechanism: '
    mod+='the installed DP batch-size synchronization precedes vision encoding, so a peer can wait inside prepare '
    mod+='for the visual producer. Lifecycle and optional lossless-encoder controls must be interpreted separately. '
    mod+='Host lifecycle timestamps are not GPU collective wait time.\n'
    save('modality_analysis.md',mod)
    pareto='# Falsification / Pareto analysis\n\n'
    pareto+='The plots include every completed discovery point, not only the fastest samples: CP256/512/1024, '
    pareto+='LP_AUTO/G4/G8/G16/G24/G48 and Full. They show actual measured TTFT versus response-window peak ITL '
    pareto+='with wave-bootstrap intervals. ReloadAmp and local prepare+finalize span shares are annotated where measured; '
    pareto+='these shares are not isolated network service fractions.\n\n'
    pareto+='![Visual measured points](../figures/pareto_visual_prefill.svg)\n\n'
    pareto+='![Matched text measured points](../figures/pareto_token_matched_text.svg)\n\n'
    pareto+=f'Frozen best LP: **{best}**; eligibility at discovery: **{d["selection"]["slo_eligible"]}**. '
    pareto+='The final ten-repeat set is confirmation, not data for selecting another group count. '
    pareto+='A remaining endpoint trade-off is not automatically an EP-specific research gap. '
    pareto+='Producer preparation, row gathers/scatters, stage rescheduling and intermediate discarded sampling '
    pareto+='are included prototype costs and must not be misreported as an intrinsic EP lower bound.\n'
    save('pareto_analysis.md',pareto)
    print('MEASURED_TABLES_WRITTEN',a.output)


if __name__=='__main__':main()
