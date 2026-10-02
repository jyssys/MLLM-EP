"""Publish completed native measurements, with explicit attribution limits.

This is a report exporter, not a latency predictor or automatic positive gate.
The immutable clean and rich runs remain the authority for every number.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json
from poc_native_lp_ep4.scripts.evidence import estimate
from poc_native_lp_ep4.scripts.tables import number, table


ROOT = Path('poc_native_lp_ep4')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--producer', type=Path, required=True)
    p.add_argument('--stages', type=Path, required=True)
    a = p.parse_args()
    d = read_json(a.metrics)
    producer = read_json(a.producer)
    stages = read_json(a.stages)
    assert producer['status'] == 'COMPLETE'
    assert d['all_logical_path_invariants_pass']
    assert len(d['current_rich_route_identity']) == 30
    assert all(r['same_exact_ids_weights'] for r in d['current_rich_route_identity'])
    final = {(r['arm'], r['modality']): r for r in d['final_summary']}
    comp = {(r['arm'], r['modality']): r for r in d['component_summary']}
    route = {(r['arm'], r['modality']): r for r in d['route_summary']}
    best = d['selection']['best_arm']
    assert best == 'LP_G8' and not d['selection']['slo_eligible']
    assert all(final[(arm, mode)]['n'] == 10 for arm in ('FULL', 'CP512', 'LP_AUTO', best)
               for mode in ('VISUAL_PREFILL', 'TOKEN_MATCHED_TEXT'))

    def save(name, text):
        path = ROOT / 'reports' / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(text)

    def measured(arm, field, mode='VISUAL_PREFILL'):
        return final[(arm, mode)][field]

    def count(arm, field, mode='VISUAL_PREFILL'):
        return route[(arm, mode)][field]

    def component(arm, field, mode='VISUAL_PREFILL'):
        return comp[(arm, mode)][field]

    serving = (a.tables / 'serving_latency.md').read_text()
    serving += '''
## Interpretation and measurement boundary

Native LP is genuinely advancing hidden/residual state across serving iterations,
not replaying layers or summing post-hoc timings. The full resident decoder runs
48 layers and emits a real token at every stage. AUTO is **G4**, following the
actual pinned upstream <=2048-token rule; the earlier expected G8 was wrong.

All visual arms retain a first-iteration peak. Increasing to G16/G24/G48 does
not monotonically reduce this producer floor and increases TTFT/stage exposure.
The secondary later-stage analysis shows that CP512 and LP can already keep
ordinary later iterations reasonably close to decode-only. Do not delete the
first burst from the primary latency statistic to manufacture a passing SLO.

TTFT starts at model-ready injection (CPU image rendering happened while both
DPs were paused). GPU vision encoding remains enabled. The reported prefill wall
is **LM-entry to first completed token**, including interleaved resident decode,
final normalization/LM head/sampling/output. It is not isolated prefill GPU time.
The injected request emits one token in timing, so its measured completion E2E
equals this model-ready TTFT; resident fixed128 makespan is in raw worker output.

Resident B8 contains four frozen images duplicated as independent requests on
the two DPs: global eight requests, **four unique images**, not eight independent
input samples. The fixed native continuation controls token/position identity;
it is not a natural task-quality evaluation. One frozen injected visual and one
matched text prompt bound the generalization domain. CIs describe measurement
repeat uncertainty, not unseen-image accuracy or all serving workloads.

Ten paired repeats span five independent engine restarts. Restart-cluster CIs
are also published in final_metrics.json; five clusters remain a small sample.
P99 over a long fixed128 run can dilute a single input burst, hence response peak
ITL and paired VSEL remain primary. No fastest-run filtering is used.
'''
    save('serving_latency.md', serving)

    ep = (a.tables / 'ep_component_analysis.md').read_text()
    ep += '\n## Per-layer cumulative critical-rank spans\n\n'
    ep += table(['Arm', 'Input', 'Dispatch/layer ms', 'Expert/layer ms',
                 'Combine/layer ms', 'MoE/layer ms'], [
        [r['arm'], r['modality'], *[f"{r[k]['mean']/48:.4f}" for k in
            ('dispatch_total_ms', 'expert_total_ms', 'combine_total_ms', 'moe_total_ms')]]
        for r in d['component_summary']])
    ep += '''
Division is by **48 original decoder layers**, including all CP chunks for a
layer. It is not a per-chunk average or an attribution of fused mixed expert
time to individual modalities. Full/text's component lane exceeded the frozen
10% overhead screen (10.302% mean TTFT) and is diagnostic **UNVALIDATED**; it is
not used to prove an EP-common residual. CP/LP strata are screened separately.

## Accumulated victim dispatch excess (diagnostic)

Paired same forced decode position, sum of local max-duration differences across
the active layers. This is a duration proxy, not a synchronized global clock,
CWT, isolated wire time or a new primary serving metric.

'''
    totals = defaultdict(lambda: defaultdict(float))
    for r in d['victim_dispatch_excess']:
        totals[(r['arm'], r['modality'], r['wave'])][r['decode_step']] += r['dispatch_excess_ms']
    groups = defaultdict(list)
    for (arm, mode, _), vs in totals.items():
        groups[(arm, mode)].append(max(vs.values()))
    ep += table(['Arm', 'Input', 'Wave-peak accumulated excess ms [95% CI]'], [
        [arm, mode, number(estimate(v))] for (arm, mode), v in groups.items()])
    ep += '''
## EP-common conclusion

**NOT_CONFIRMED**, not a claim that EP is free. LP preserves Full-like active
prefill expert execution, whereas CP512 fragments it. The very long first
prepare interval can be peer-producer readiness, because DP padding exchange
occurs before the visual encoder. It is not measured intrinsic non-preemptible
A2A service/MNCD. No isolated network/readiness subtraction or architecture-
matched non-EP control is available here. The separate cached producer diagnostic
is described in modality_analysis.md. Large local communication span share alone
does not establish the specified common EP lower-bound gate.

Hardware DRAM/HBM reads, wire metadata/padding bytes, isolated CWT and Nsight
profiling are **NOT_RUN**, never filled with estimates. ExpertReloadAmp and the
9MiB/expert working-set sum are structural proxies, not measured HBM traffic.
'''
    save('ep_component_analysis.md', ep)

    mod = (a.tables / 'modality_analysis.md').read_text()
    mod += '\n## Stable-denominator serving penalty\n\n'
    mod += table(['Arm', 'Paired visual-minus-text peak ITL (%) [95% CI]', 'Pairs'], [
        [arm, number(fields['peak_itl_ms_percent']), fields['peak_itl_ms_percent']['n']]
        for arm, fields in d['final_modality_deltas'].items()])
    geo = {(r['arm'], r['modality']): r for r in d['route_geometry_waves']}
    gv, gt = geo[('LP_G8', 'VISUAL_PREFILL')], geo[('LP_G8', 'TOKEN_MATCHED_TEXT')]
    mod += (f"\nFrozen G8 logical dispatch across all 48 layers is **{gv['logical_hidden_dispatch_bytes']/1e9:.6f} GB visual** "
            f"versus **{gt['logical_hidden_dispatch_bytes']/1e9:.6f} GB text**. Destination fanout is "
            f"{gv['mean_distinct_destination_ranks_per_prompt_token']:.4f} versus {gt['mean_distinct_destination_ranks_per_prompt_token']:.4f}; "
            f"mean layer assignment max/mean is {gv['mean_layer_rank_assignment_max_mean']:.4f} versus {gt['mean_layer_rank_assignment_max_mean']:.4f}. "
            'Visual is not larger in these logical traffic/skew counters. These actual measurements argue against explaining '
            'the large victim penalty solely as greater visual A2A payload/fanout.\n')
    mod += '\n## Actual producer lifecycle (separate light trace)\n\n'
    mod += table(['Arm', 'Input', 'Admission→first worker ms', 'Vision encoder CUDA ms',
                  'Initial input pipeline CUDA ms', 'Arrival→LM entry ms'], [
        [r['arm'], r['modality'], *[number(r[k]) for k in
            ('admission_to_first_worker_ms', 'injection_vision_encoder_cuda_ms',
             'initial_input_pipeline_cuda_ms', 'arrival_to_lm_entry_ms')]]
        for r in d['lifecycle_summary']])
    mod += '''
The roughly 80ms host delivery/setup/executor interval and roughly 30ms vision
encoder duration occur once for the injected image, not once per LP stage.
These different clock intervals must not be added and equated to a GPU global
critical path. The host segment does not identify an exact IPC/serialization
routine; that attribution would require another profiler. The installed source
does show that the other DP can reach DeepEP prepare before visual input becomes
ready, making the first prepare span a peer-readiness diagnostic.

Visual and matched text have almost equal LM row counts. Actual source-SP,
destination-deduplicated logical traffic/fanout/skew is included above and as
per-rank matrices in final_metrics.json; it does not explain the observed first
producer floor merely by a 4-token difference. This is a **modality-common
input-producer/runtime residual in the tested harness**, not proof of an intrinsic
MLLM-specific wire-communication mechanism.

## Lossless vision-output control (secondary, three repeats)

This diagnostic reuses exact frozen CPU processor tensor/grid hashes and the
unchanged GPU vision+DeepStack output. It is not a production caching policy or
a clean hot-path optimization. Exact CPU input hashing and clone lookup remain
included in the measured request path. Cached and uncached Full have identical
prefill hidden/logits and unforced eight-token continuation in six comparisons;
cached Full/LP native correctness passes three repeats for both inputs.

'''
    mod += table(['Arm', 'Input', 'Cached secondary TTFT ms', 'Peak VSEL ms', 'Peak ITL ms'], [
        [r['arm'], r['modality'], number(r['ttft_ms']), number(r['peak_vsel_ms']),
         number(r['peak_itl_ms'])] for r in producer['clean_summary'] if r['arm'] != 'DECODE_ONLY'])
    mod += table(['Arm', 'Input', 'Cached prepare span max ms', 'Cached expert total ms'], [
        [r['arm'], r['modality'], number(r['max_dispatch_quantum_ms']), number(r['expert_total_ms'])]
        for r in producer['component_summary']])
    mod += table(['Arm', 'Input', 'Cached admission→worker ms', 'Cached encoder-wrapper CUDA ms'], [
        [r['arm'], r['modality'], number(r['admission_to_first_worker_ms']),
         number(r['injection_vision_encoder_cuda_ms'])] for r in producer['lifecycle_summary']])
    mod += '''
Use these actual diagnostic outcomes, including negative/no-improvement results,
not a hypothetical cache speedup. Cache-hit events are legacy wave/rank-only:
resident hits alone cannot prove the injection was cached. Exact numerical
identity and the injection-specific encoder-wrapper durations are independent
checks. Removing the vision encoder can shorten the local first-prepare span
while CPU hashing/setup still delays victim completion. These three-repeat
secondary runs are not pooled with or substituted for the ten-repeat primary.
'''
    save('modality_analysis.md', mod)

    pareto = (a.tables / 'pareto_analysis.md').read_text().replace('../figures/', '../figures/release_v8/')
    pareto += '''
## What is and is not falsified

Native LP solves the **CP-specific expert reload** problem: every LP group count
has ReloadAmp 1.000x and Full-like active mixed expert execution. AUTO/G4 also
has substantially lower TTFT than finer LP groups. G8 was frozen from the noisy
three-repeat discovery minimum; final G8 is not better than AUTO, and we do not
reselect a winner or claim group count must monotonically improve peak TBT.

No tested visual point passes the complete response-window 1.5x decode-median
proxy, owing to the first producer iteration. Later stages alone generally do.
This leaves a **tested visual-input producer Pareto residual**, not a demonstrated
communication-only slicing opportunity. Native row gathers/scatters, intermediate
discarded sampling and repeated useful resident decode contribute to fine-group
TTFT. Their overhead cannot be generalized as an intrinsic LP or EP deficiency.

Thus retire the prior blanket claim that LP cannot maintain expert efficiency,
and do not promote communication-only chunking from this campaign. Investigate
lossless asynchronous multimodal input preparation/delivery first, and validate
outside this controlled harness before claiming a general serving bottleneck.
There is no new adaptive scheduler, communication slicing or model approximation.
'''
    save('pareto_analysis.md', pareto)

    verdict = {
        'schema': 'native-lp-final-verdict-v1',
        'implementation_scope': 'CONTROLLED_NATIVE_LP_EP4_HARNESS',
        'overall_verdict': 'MODALITY_COMMON_RESIDUAL_FOUND',
        'scope_qualification': 'TESTED_MULTIMODAL_INPUT_PRODUCER_RUNTIME_PATH; NOT_INTRINSIC_VISUAL_WIRE_COST',
        'native_correctness': 'PASS', 'decode_every_iteration': 'PASS',
        'ep_common_residual': 'NOT_CONFIRMED',
        'modality_common_residual': 'SUPPORTED_TESTED_INPUT_PRODUCER_PATH',
        'common_ep_component': 'NONE_ISOLATED_AS_AN_INTRINSIC_EP_LOWER_BOUND',
        'common_modality_component': 'INITIAL_MULTIMODAL_DELIVERY_SETUP_AND_VISION_PRODUCER_READINESS',
        'pareto_gap': 'TESTED_VISUAL_INPUT_PRODUCER_GAP_ONLY; NO_COMMUNICATION_METHOD_PROMOTION',
        'frozen_best_lp': best,
        'best_lp_slo_eligible': False,
        'current_rich_route_comparisons': 30,
        'logical_path_wave_checks': len(d['route_checks']),
        'next_action': 'Retire LP expert-fragmentation claim; study lossless multimodal producer isolation before any new communication method.',
        'limits': d['limits'] + [
            'One frozen visual injection / one token-matched text; B8 has four unique resident images.',
            'Full/text component overhead exceeds 10%; its decomposition is UNVALIDATED.',
            '288 inactive trace-window partial frames retained; zero partial active-prefill frames.',
            'Optional non-EP and HBM/wire/CWT isolation NOT_RUN.',
            'Lossless cache diagnostic includes exact CPU hashing and is secondary, not an optimized speedup.'
        ]}
    atomic_json(ROOT / 'artifacts/final_metrics.json', d)
    atomic_json(ROOT / 'artifacts/final_verdict.json', verdict)
    summary = '# Native Layered Prefill on TRUE EP4 — final discovery\n\n'
    summary += '**MODALITY_COMMON_RESIDUAL_FOUND**, scoped to the tested visual input-producer/runtime path. '
    summary += '**No intrinsic common EP collective bottleneck is established; do not promote a new communication method.**\n\n'
    summary += 'Actual native LP is correct; resident decode emits a token through all 48 layers on every stage. '
    summary += 'Full/LP hidden and logits are bit-exact in 30 acceptance comparisons, with eight unforced tokens identical. '
    summary += 'A fresh pooled v7 check adds 12 passing comparisons. Current rich prefill route IDs/weights match in 30 comparisons.\n\n'
    summary += table(['Visual input arm', 'TTFT ms [95% CI]', 'Peak VSEL ms [95% CI]',
                      'P99 ITL ms [95% CI]', 'ReloadAmp', 'Active mixed expert total ms'], [
        [arm, number(measured(arm, 'ttft_ms')), number(measured(arm, 'peak_vsel_ms')),
         number(measured(arm, 'p99_itl_ms')), number(count(arm, 'expert_reload_amp'), 3),
         number(component(arm, 'expert_total_ms'))]
        for arm in ('FULL', 'CP512', 'LP_AUTO', best)])
    summary += '\n10 paired clean repeats/arm/input across five engine restarts; components/routes use separate three-repeat lanes.\n\n'
    summary += 'CP512 versus text adds **'+number(d['final_modality_deltas']['CP512']['peak_vsel_ms_delta'])
    summary += ' ms** peak VSEL; frozen G8 adds **'+number(d['final_modality_deltas'][best]['peak_vsel_ms_delta'])+' ms**. '
    summary += 'The extra peak is concentrated in the first producer iteration. Subsequent LP stages retain low expert reload '
    summary += 'and reasonable per-token latency. G4/8/16/24/48 all execute natively; increasing groups does not remove '
    summary += 'the initial visual producer floor. CP expert fragmentation is real but **CP-specific**, not common EP evidence.\n\n'
    summary += 'The secondary lossless producer diagnostic and lifecycle clocks are in [modality_analysis.md](modality_analysis.md). '
    summary += 'A prepare interval includes peer readiness and is **not isolated A2A service time**. '
    summary += 'No post-hoc replay, simulator, full benchmark rerun, model approximation or new method is used.\n\n'
    summary += 'All measured Pareto points are in [pareto_analysis.md](pareto_analysis.md); '
    summary += 'the positive residual does not overturn the native LP efficiency success. '
    summary += 'The sample pool is narrow; shared-fabric foreign load and prototype overhead constrain generalization.\n\n'
    summary += '## Required final fields\n\n```text\n'
    fields = {
        'GPU_SET': '0,1,4,5; H100 80GB; NV18/NVSwitch (UUIDs in baseline_readiness.md)',
        'MODEL': 'Qwen3-VL-30B-A3B-Instruct @ 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c; BF16; d2048/m768/E128/K8/L48',
        'TOPOLOGY': 'TP2-DP2-EP4-PP1; linear 32 experts/rank; source SP, no counted duplicate TP paths',
        'DEEPEP_BACKEND': 'deepep_high_throughput + sparse Triton; DeepEP 1.2.1+73b6ea4',
        'UPSTREAM_LAYERED_COMMIT': '053f80e5201a7c0ab56e468e3d578907a2ca9cc3 (official TP-only code)',
        'LP_IMPLEMENTATION_SCOPE': verdict['implementation_scope'],
        'NATIVE_LP_CORRECTNESS': 'PASS; exact hidden/logits, first token +8 continuation; rich route hashes PASS',
        'DECODE_CONTINUES_EVERY_ITERATION': 'PASS; full48 layers / one real token per resident / per stage',
        'PREFILL_LAYER_GROUP_EXECUTION_VERIFIED': 'PASS; persistent hidden/residual; 48 prefill layers exactly once',
        'CP512_TTFT_MS': number(measured('CP512', 'ttft_ms')),
        'CP512_PEAK_VSEL_MS': number(measured('CP512', 'peak_vsel_ms')),
        'CP512_ITL_P99_MS': number(measured('CP512', 'p99_itl_ms')),
        'CP512_EXPERT_RELOAD_AMP': number(count('CP512', 'expert_reload_amp'), 4),
        'CP512_DISPATCH_MS_PER_LAYER': f"{component('CP512', 'dispatch_total_ms')['mean']/48:.4f}; local prepare span, not isolated wire",
        'LP_AUTO_STAGE_COUNT': 4,
        'LP_AUTO_TTFT_MS': number(measured('LP_AUTO', 'ttft_ms')),
        'LP_AUTO_PEAK_VSEL_MS': number(measured('LP_AUTO', 'peak_vsel_ms')),
        'LP_AUTO_ITL_P99_MS': number(measured('LP_AUTO', 'p99_itl_ms')),
        'LP_AUTO_EXPERT_RELOAD_AMP': number(count('LP_AUTO', 'expert_reload_amp'), 4),
        'LP_AUTO_DISPATCH_MS_PER_ACTIVE_LAYER': f"{component('LP_AUTO', 'dispatch_total_ms')['mean']/48:.4f}; includes peer readiness",
        'LP_BEST_GROUP_COUNT': '8; frozen discovery minimum, NOT_SLO_ELIGIBLE; not final-selected winner',
        'LP_BEST_TTFT_MS': number(measured(best, 'ttft_ms')),
        'LP_BEST_PEAK_VSEL_MS': number(measured(best, 'peak_vsel_ms')),
        'LP_BEST_ITL_P99_MS': number(measured(best, 'p99_itl_ms')),
        'LP_BEST_EXPERT_RELOAD_AMP': number(count(best, 'expert_reload_amp'), 4),
        'LP_BEST_DISPATCH_MS_PER_ACTIVE_LAYER': f"{component(best, 'dispatch_total_ms')['mean']/48:.4f}; includes peer readiness",
        'CP_VISUAL_MINUS_TEXT_DISPATCH_PCT': number(d['component_modality_deltas']['CP512']['dispatch_total_ms_percent']) + '; n3 local-span diagnostic',
        'LP_VISUAL_MINUS_TEXT_DISPATCH_PCT': number(d['component_modality_deltas'][best]['dispatch_total_ms_percent']) + '; n3 local-span diagnostic',
        'CP_VISUAL_MINUS_TEXT_VSEL_PCT': number(d['final_modality_deltas']['CP512']['peak_vsel_ms_percent']) + '; n10, text denominator near zero, prefer absolute delta',
        'LP_VISUAL_MINUS_TEXT_VSEL_PCT': number(d['final_modality_deltas'][best]['peak_vsel_ms_percent']) + '; n10, text denominator small, prefer absolute delta',
        'COMMON_EP_COMPONENT': verdict['common_ep_component'],
        'COMMON_MODALITY_COMPONENT': verdict['common_modality_component'],
        'PARETO_GAP_EXISTS': verdict['pareto_gap'],
        'EP_COMMON_RESIDUAL': verdict['ep_common_residual'],
        'MODALITY_COMMON_RESIDUAL': verdict['modality_common_residual'],
        'OVERALL_VERDICT': verdict['overall_verdict'],
        'NEXT_ACTION': verdict['next_action'],
        'BRANCH': 'flashvep/native-layered-prefill-ep4-discovery',
        'COMMIT': 'Publication evidence SHA recorded in artifacts/release.json after evidence freeze',
    }
    summary += '\n'.join(f'{k}: {v}' for k, v in fields.items()) + '\n```\n'
    save('final_summary.md', summary)
    atomic_json(ROOT / 'artifacts/report_provenance.json', {
        'metrics_source': str(a.metrics), 'metrics_sha256': hashlib.sha256(a.metrics.read_bytes()).hexdigest(),
        'producer_source': str(a.producer), 'producer_sha256': hashlib.sha256(a.producer.read_bytes()).hexdigest(),
        'secondary_stage_source': str(a.stages),
        'source': 'Completed physical native executions only; no simulated or fabricated rows',
        'report_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'decision_fields': fields})
    print('FINAL_REPORTS_WRITTEN', verdict['overall_verdict'])


if __name__ == '__main__':
    main()
