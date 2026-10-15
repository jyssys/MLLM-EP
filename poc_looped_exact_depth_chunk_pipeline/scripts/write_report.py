"""Write the final bounded report from completed evidence, not partial scores."""
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    quality=read(ROOT/'results/native_quality_r4.json')
    assert len(quality['final_lanes'])==3 and all(r['collection_complete'] for r in quality['final_lanes'].values())
    assert read(ROOT/'results/quality_native_v3/guard_status.json')['success']
    assert read(ROOT/'results/physics_completion.json')['success']
    reference=read(ROOT/'results/reference_components_v1/reference_component_correctness.json')
    transfers=read(ROOT/'results/stage_transfers_v1/transfers.json')
    bounds=read(ROOT/'results/schedule_bounds.json')
    native=[r for r in read(ROOT/'results/baseline_summary.json') if r['TP']==4 and r['R']==4 and r['chunk']==0]
    execution=read(ROOT/'results/native_execution_audit.json')
    cap_path=ROOT/'results/cap_prefix_reproducibility.json'
    verdict='QUALITY_BASELINE_WEAK' if quality['quality_gate']!='QUALITY_BASELINE_PASS' else (
        'CORRECTNESS_BLOCKED' if not reference['all_component_pass'] else 'GENERIC_PP_EQUIVALENT')
    decision=dict(verdict=verdict,native_quality_gate=quality['quality_gate'],dependency_gate='GENERIC_PP_EQUIVALENT',
        component_reference_gate='PASS' if reference['all_component_pass'] else 'FAIL_UNDER_FROZEN_TOLERANCE',
        component_reference_verdict='PASS' if reference['all_component_pass'] else 'CORRECTNESS_BLOCKED',
        official_reference='REFERENCE_PROTOCOL_NOT_IDENTICAL',integrated_generic_PP='NOT_RUN',
        candidate_runtime='NOT_RUN',candidate_numerical_correctness='NOT_RUN',candidate_24_natural_parity='NOT_RUN',
        candidate_benchmark_quality='NOT_RUN',measured_candidate_E2E_gain='NOT_RUN',
        headroom_status=bounds['status'],next_action='Do not build a loop-specific framework from this DAG. Resolve native complete-output quality first if its gate is weak.',
        native_CP_sweep='RUN' if bounds.get('models') else 'NOT_RUN_QUALITY_GATE',
        branch='flashvep/looped-exact-depth-chunk-poc',parent=read(ROOT/'results/environment.json')['parent'])
    write(ROOT/'results/verdict.json',decision)
    lines=['# Exact recurrent-depth × sequence-chunk pipeline', '',f'Final verdict: `{verdict}`.', '',
        '## Decision', '',
        'The dependency graph admits a mathematically legal chunk schedule.',
        'It is isomorphic to generic PP on the tied-unrolled 192-layer operation graph.',
        'No independent recurrent-specific scheduling or memory advantage passes that comparison.',
        'No distributed candidate serving engine or integrated generic PP engine runs in this campaign.',
        'We claim no measured pipeline E2E speedup or candidate-quality PASS.', '',
        f'Native quality gate: `{quality["quality_gate"]}`. Component reference gate: `{decision["component_reference_gate"]}`.', '',
        f'The independent novelty verdict is `GENERIC_PP_EQUIVALENT`. The reference-executor verdict is `{decision["component_reference_verdict"]}`.',
        'Mathematical graph equivalence does not establish BF16 execution parity.', '',
        '## Model, runtime, and physical devices', '',
        'The checkpoint is ByteDance/Ouro-2.6B-Thinking at `f1edd81e7ac41355db670500ceaf204e0f73af68`.',
        'The native engine is vLLM 0.26.0+cu129 with PyTorch 2.11.0+cu129 and Transformers 5.16.1.',
        'It uses BF16, FlashAttention3, ordinary CUDA Graphs, and native TP4.',
        'Only physical H100 GPUs4–7 execute kernels. Logical CUDA ranks0–3 map to those devices in order.',
        'The topology audit records NV18 between every assigned GPU pair. Foreign processes remain untouched.',
        'Each native TP4 rank has 192 cache slots and four local KV heads.',
        'The loaded model has 48 physical layers and four passes. The card says 24 layers.',
        'We preserve the loaded pinned model. We do not change its layer count to fit the card.', '',
        'The config permits 65,536 positions. This campaign caps total positions at 32,768, consistent with the documented Thinking context.',
        'Prefix caching, asynchronous scheduling, and EP remain disabled.', '',
        'See [architecture provenance](model_architecture_vs_reference.md), [environment](results/environment.json), and [runtime audit](results/audit.json).', '',
        '## Native public quality, not forced-token accuracy', '',
        'All quality outputs are real natural generation. The quality lane keeps EOS enabled.',
        'The official tokenizer chat template uses `add_generation_prompt=True`.',
        'R remains four. The early-exit threshold remains 1.0. Native vLLM executes all four passes.',
        'Published-style sampling uses temperature1.0 and top-p0.7. Seeds are 41015, 41016, and 41017.',
        'The greedy lane uses temperature0. Quality batches contain at most eight requests.',
        'Every problem, answer, prompt, and seed freezes before quality output.', '',
        '| Dataset/lane | Cap | Correct completed / total | Exact-answer score, problem-bootstrap 95% CI | EOS / total | Parsed / total |',
        '|---|---:|---:|---|---:|---:|']
    for row in sorted(quality['all_caps'],key=lambda r:(r['dataset'],r['lane'],r['cap'])):
        lo,hi=row['problem_bootstrap_95CI']
        lines.append(f'| {row["dataset"]}/{row["lane"]} | {row["cap"]} | {row["correct"]}/{row["total"]} | '
            f'{100*row["score"]:.2f}% [{100*lo:.2f}%, {100*hi:.2f}%] | {row["natural_complete"]}/{row["total"]} | {row["parsed"]}/{row["total"]} |')
    lines += ['', 'The final analysis uses the largest complete preregistered cap in each lane.',
        'A cap increase repeats the entire lane. It does not select only failed or favorable problems.',
        'Capped generations remain in the denominator and cannot count as completed correct answers.',
        'The raw ledger separately records matching extracted answers inside capped outputs. That diagnostic is not completed accuracy.',
        'AIME has 30 independent problems. The bootstrap retains all seeds of each selected problem together.',
        'MATH100 is a subject-stratified subset. It is not the full MATH-500 score.', '',
        '### Final-cap seed results', '', '| Lane | Seed | Correct / total |', '|---|---:|---:|']
    for name,row in quality['final_lanes'].items():
        for seed,value in row['by_seed'].items():lines.append(f'| {name} | {seed} | {value["correct"]}/{value["total"]} |')
    lines += ['', '### Final-cap output diagnostics', '',
        '| Lane | Mean tokens | Median tokens | P95 tokens | Capped / total | Mean repeated 16-gram fraction |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in quality['final_lanes'].items():
        lines.append(f'| {name} | {row["mean_generated_tokens"]:.1f} | {row["median_generated_tokens"]:.1f} | '
            f'{row["p95_generated_tokens"]:.1f} | {row["capped"]}/{row["total"]} | {row["mean_repeated_16gram_fraction"]:.4f} |')
    lines += ['', 'The repetition statistic counts duplicate overlapping 16-token sequences.',
        'It is a descriptive diagnostic, not a quality judge or a model-selection rule.']
    lines += ['', '### Comparison with the authors', '',
        'The authors report AIME2024 pass@1 64.7%, AIME2025 50.3%, and OlympiadBench 76.44%.',
        'Their in-house prompt/harness and LLM judge do not match this public deterministic evaluator.',
        'The card example omits explicit do_sample and top-k settings. This run explicitly samples with top-k disabled.',
        'The comparison is `REFERENCE_PROTOCOL_NOT_IDENTICAL`, not an official-score reproduction.',
        'The official repository address returned HTTP404 during this audit. Its complete evaluation recipe remains unavailable.',
        'The project page links to LoopLM. That repository describes a reimplementation, not the original private evaluation harness.',
        'A low bounded score with substantial truncation does not establish that the pretrained model lacks the reported capability.',
        'No prompt, temperature, top-p, seed, recurrence, precision, or answer key changes based on observed answers.',
        'Only the preregistered complete-lane cap rule changes the output budget.', '',
        'The exact AIME grader accepts only a final integer in [0,999].',
        'MATH uses Math-Verify0.8.0 with LaTeX extraction, strict equivalence, and two-second limits.',
        'All 100 frozen MATH references pass the parser self-check. Preset equivalence and wrong-answer controls also pass.',
        'No manual answer adjudication changes a score. Two assistant inspections check EOS versus truncation only.',
        'The first two execution attempts fail before benchmark generation because of tokenizer preflight checks.',
        'The corrected path preserves every frozen token sequence. Two secondary prompts have different Transformers5 space tokenization but identical decoded prompt strings.',
        'The seed check preserves both short same-batch repetitions. It is not a full-output universal determinism proof.', '',
        'See [quality protocol](quality_protocol.md), [raw scores](results/quality_generations.csv), and [native quality JSON](results/native_quality_r4.json).', '',
        '### Cross-cap prefix diagnostic', '']
    if cap_path.exists():
        cap_check=read(cap_path)
        lines += [f'{cap_check["same_prefix_count"]}/{cap_check["total"]} paired outputs preserve their common token prefix across the two caps.',
            'All 120 AIME pairs match. Twelve of 100 MATH greedy pairs diverge within their common prefix.',
            'This diagnostic changes max_tokens. It is not an identical full-configuration repeat.',
            'The cause remains unresolved. The MATH score change cannot be attributed only to the larger budget.',
            'No full-output reproducibility PASS follows from the short seed check.', '',
            'See [per-problem first divergence](results/cap_prefix_reproducibility.json).', '']
    else:lines += ['Cross-cap prefix diagnostic: `NOT_RUN`.', '']
    lines += [
        '## Exact DAG and generic PP', '',
        'Task F[r,j] includes a full 48-layer pass and its boundary norm.',
        'It needs the preceding-depth hidden state for chunk j and committed same-depth prefix KV.',
        'Intra-chunk masks use original causal positions. RoPE positions never restart at a chunk boundary.',
        'The macro graph is a sufficient whole-stack schedule. It is not a minimal operator schedule.',
        'The micro graph separates KV production, publication, layer output, and the four native pass norms.',
        'Flattening loop/layer to `48*r+layer` exactly preserves its edge set.',
        'An independent generic tied-unrolled construction matches that relabeled graph.',
        'Finer operator streaming therefore also exists in generic PP. It is not a recurrence-only opportunity.', '',
        'Each depth stage owns its KV. Interstage prefill traffic contains hidden activations, not remote KV reads.',
        'For a single new output token, all four depths remain sequential. Unknown future tokens cannot fill the pipeline without speculation.', '',
        'See [dependency contract](dependency_contract.md), [edge-set audit](results/dependency_gate.json), and [ownership](stage_memory_ownership.md).', '',
        '## Fresh optimized native TP4 baseline', '',
        'This hardware-only lane uses B1 and 64 frozen output IDs. It is not the quality lane.',
        'Each configuration uses three warmups per prompt length. Each request then uses three technical repeats.',
        'The analysis averages repeats before resampling requests with 5,000 draws.',
        'Prefill means the native scheduled-to-first-token envelope. It is not attention-only time.',
        'Cached TPOT uses 63 intervals. Clean E2E includes output handling after the loaded engine.', '',
        '| Prompt tokens | n | Prefill ms | Cached TPOT ms | Clean E2E ms, 95% CI | Peak allocated GiB/rank |', '|---:|---:|---:|---:|---|---:|']
    for row in sorted(native,key=lambda r:r['P']):
        x=row['E2E_ms'];lines.append(f'| {row["P"]} | {row["n_requests"]} | {row["prefill_ms"]["mean"]:.3f} | '
            f'{row["TPOT_ms"]["mean"]:.3f} | {x["mean"]:.3f} [{x["low"]:.3f}, {x["high"]:.3f}] | {row["peak_HBM_bytes"]["mean"]/2**30:.3f} |')
    lines += ['', 'Historical absolute times never supply this baseline.',
        'Peak allocated memory is the request maximum across ranks. It is not the same as the device-wide nvidia-smi memory counter.',
        f'The separate trace records four actual forwards on each of four ranks. Its 192-slot execution-count check is {execution["all_count_checks_pass"]}.',
        'Each rank records three actual CUDA Graph replays and four sampling calls.',
        'The host counter records five execute_model calls. The GPU trace contains four active forwards, not five 192-layer executions.',
        f'Profiler observer overhead is {execution["observer_percent"]:.2f}%. Timing attribution is `{execution["timing_attribution"]}`.',
        'The execution audit counts every observed forward. It does not multiply the final graph replay by output length.',
        'The native PP class does not provide the required recurrent-depth stage interface.',
        'A PP option or physical-layer partition does not establish exact 192-layer pipeline serving.', '',
        '## Physical activation transfer', '',
        '| Chunk rows | One-edge hidden bytes | Maximum PP4-edge median ms | Duplicated PP2×TP2 boundary median ms |', '|---:|---:|---:|---:|']
    for B in [256,512,1024,2048,4096]:
        pp=[r['median_ms'] for r in transfers if r['rows']==B and r['mode'].startswith('PP4_edge')]
        tp=next(r for r in transfers if r['rows']==B and r['mode']=='PP2_TP2_duplicate_boundary')
        lines.append(f'| {B} | {B*2048*2} | {max(pp):.4f} | {tp["median_ms"]:.4f} |')
    lines += ['', 'These are physical NCCL send/receive paths on the four assigned GPUs.',
        'The interval includes device completion and the final coordination barrier. It is not pure transport bandwidth.',
        'The benchmark uses three warmups and 30 repeats. Those repeats are not independent serving requests.',
        'It does not measure an integrated overlapped pipeline or assert free transfer.', '',
        'The transfer sweep uses a preset order. Nonmonotonic medians include host/world-barrier overhead.',
        'They do not establish a transport-size scaling law.', '',
        '## Stage economics and E2E bounds', '',f'Cost-model status: `{bounds["status"]}`.', '']
    if bounds.get('models'):
        lines += ['The component sweep measures real pretrained one-pass stages with native attention and correct accumulated prefix KV.',
            'Those R1 physics probes are not quality-equivalent R4 inference.',
            'Later-depth activation costs use a documented shape-based projection. Integrated contention and cache transition remain unvalidated.',
            'Every PP forecast is also the generic tied-PP forecast. There is no distinct B4 method.', '',
            '| P | Best modeled layout/chunk | Prefill reduction, 95% CI | Optimistic E2E reduction, 95% CI |', '|---:|---|---|---|']
        for P in [1024,8192,16384,32704]:
            points=[r for r in bounds['models'] if r['P']==P]
            if not points:continue
            row=max(points,key=lambda r:r['optimistic_E2E_reduction_percent']['mean']);p=row['prefill_reduction_percent'];e=row['optimistic_E2E_reduction_percent']
            lines.append(f'| {P} | PP{4//row["TP_per_stage"]}×TP{row["TP_per_stage"]}, chunk{row["chunk"]} | '
                f'{p["mean"]:.2f}% [{p["low"]:.2f}%, {p["high"]:.2f}%] | {e["mean"]:.2f}% [{e["low"]:.2f}%, {e["high"]:.2f}%] |')
        lines += ['', 'These are calibrated forecasts, not measured speedups.',
            'The optimistic E2E ledger grants unchanged native decode and a free stage-KV-to-TP4 transition.',
            'That transition would redistribute nonzero cache bytes. The optimistic result is not an executable method.',
            'A numerical CI over three component requests does not cover every modeling or engine uncertainty.']
    else:
        lines += ['The native-quality gate stops the broad one-pass and native-chunk sweep.',
            'One-pass stage costs, the CP sweep, calibrated PP improvements, and candidate E2E improvements are `NOT_RUN`.',
            'The tiny reference and activation-transfer diagnostics do not replace that missing physical comparison.',
            'No structural DAG headroom becomes measured speedup.']
    if not bounds.get('models'):
        lines += ['', 'The chunk-size sensitivity, physical pipeline Gantt, and generated-length headroom sweeps are `NOT_RUN`.',
            'Their absence follows the native-quality and generic-equivalence stop gates. Missing values do not mean zero headroom.']
    lines += ['', '## Component correctness and conditional full-model tests', '',
        f'The official BF16 HF reference executes {len(reference["component_results"])} full-versus-chunk comparisons on two frozen inputs.',
        f'Its frozen component gate is `{decision["component_reference_gate"]}`.',
        'Frozen limits are 0.125 for maximum absolute logit error and 0.01 for hidden and per-slot KV relative L2.',
        'Selected greedy top-1 IDs must also match.',
        'Every task checks all 48 same-depth prefix-cache lengths. Completed tasks preserve 192 independent slots.',
        'The reference is single-GPU SDPA. Its wall is not an optimized serving benchmark.',
        'Cross-engine native TP4 comparisons preserve full logits and selected written KV under identical token prefixes.',
        'BF16 cross-engine numerical differences receive explicit records. Short probes cannot certify complete natural-output parity.', '',
        '| Reference input | Chunk | Max absolute logit difference | Hidden relative L2 | Worst-slot KV relative L2 | Selected top-1 equal | Frozen gate |',
        '|---|---:|---:|---:|---:|---|---|']
    for row in reference['component_results']:
        worst=max(max(v['K_relative_L2'],v['V_relative_L2']) for v in row['KV'])
        lines.append(f'| {row["request_id"]} | {row["chunk"]} | {row["logits_max_abs"]:.6f} | '
            f'{row["hidden_relative_L2"]:.6f} | {worst:.6f} | {row["all_selected_greedy_top1_equal"]} | {row["PASS"]} |')
    if not reference['all_component_pass']:
        lines += ['', 'The tested chunk executor does not satisfy the frozen BF16 numerical contract.',
            'Matching selected top-1 IDs cannot override hidden/KV/logit failures.',
            'The recorded errors grow across logical depth. Their exact numerical cause remains unresolved.',
            'The logarithmic KV plot displays zero errors at 1e-7. The raw JSON retains their exact zero values.',
            'This campaign does not prove that every exact chunk implementation must fail. It does not relax limits or promote this executor.']
    lines += ['', '| Native/HF probe | Four teacher-aligned top-1 IDs equal | Independent four-token output equal | Full-logit relative L2 | Hidden relative L2 | Sampled-KV relative L2 |',
        '|---|---|---|---:|---:|---:|']
    for row in reference['cross_engine']:
        lines.append(f'| {row["problem_id"]} | {row["same_four_greedy_ids"]} | {row["independent_four_token_natural_equal"]} | {row["full_logits_relative_L2"]:.6f} | '
            f'{row["last_hidden_relative_L2"]:.6f} | {row["sampled_KV_relative_L2"]:.6f} |')
    lines += ['', 'Native KV registration and HF cache indices use different iteration orders.',
        'The comparison maps each native registered cache name to its logical index before comparing tensor values.', '',
        '| Test | Status |', '|---|---|',
        f'| Native R4 public quality | {quality["quality_gate"]} |',
        f'| HF exact-chunk component check | {decision["component_reference_gate"]} |',
        '| Integrated candidate numerical/KV test | NOT_RUN |',
        '| Candidate 24-request full greedy parity | NOT_RUN |',
        '| Candidate AIME/MATH quality | NOT_RUN |',
        '| Integrated generic PP timing | NOT_RUN |',
        '| Measured candidate E2E improvement | NOT_RUN |', '',
        'No quality-preserving candidate speedup claim follows from these component probes.',
        'The earlier retained-graph natural-output mismatch remains preserved in the preceding PoC.',
        'This campaign does not reinterpret it as a universal parity PASS.', '',
        '## Required decision checklist', '',
        '1. The model/runtime pins, GPU UUIDs, and NV18 topology are in the environment and ownership records.',
        '2. Same-chunk preceding-depth hidden states and contiguous same-depth prefix KV constrain overlap.',
        '3. The proposed graph is identical to generic tied-unrolled PP, including native pass norms.',
        '4. No shared-weight mechanism saves work beyond the matched generic graph.',
        '5. The small official reference reports its frozen numerical/cache gate separately from graph validity. The current executor does not establish BF16 parity.',
        '6. Runnable macro tasks follow the diagonal frontier. The dependency-only ideal has M+3 slots for four equal stages.',
        '7. One-pass stage costs exist only if the quality-gated broad sweep runs. Missing values are unknown, not zero.',
        '8. Actual BF16 hidden transfers use `rows*2048*2` bytes per boundary edge. The component benchmark measures their full-path times.',
        '9. Stage-local KV stays local during prefill. Native TP4 decode would need a real cache transition.',
        '10. Only already-produced activations can transfer during independent work on another chunk.',
        '11. Startup/drain and resource constraints remain in the schedule. A dependency-only utilization is not measured GPU utilization.',
        '12. Layer-level publication dependencies exist. The generic micro-graph retains those same edges.',
        '13. Integrated generic PP does not run. A model cannot establish that it beats optimized native chunked TP4.',
        '14. No measured candidate E2E headroom exists. Any available forecast has explicit transition and contention limits.',
        '15. Weights, logical KV, allocated cache pools, and workspace remain separate memory accounts.',
        '16. The native public benchmark produces real outputs. Published protocol reproduction and candidate quality remain distinct gates.',
        '17. No independent advantage survives the generic PP equivalence objection.',
        f'18. Final verdict is `{verdict}`. Do not implement a loop-specific framework from these results.', '',
        '## Evidence and next action', '',
        'All benchmark outputs, input IDs, seeds, caps, and failure logs remain preserved.',
        'Existing dirty/staged changes and earlier PoCs remain unchanged.',
        'Git includes the bounded numerical arrays and heavy traces. A SHA256 manifest also records external backup copies.',
        'The scoped GPU guard restores project burners. No foreign process receives a signal.',
        'If native quality remains weak, first obtain an adequate complete-output evaluation protocol. Do not tune the model to match a reference score.',
        'Even a complete quality baseline would not remove the generic PP equivalence.',
        'See [reproduction](reproduce.md), [timing contract](timing_contract.md), [prior boundary](prior_overlap.md), and [verdict JSON](results/verdict.json).', '',
        'Branch: `flashvep/looped-exact-depth-chunk-poc`.',
        'The delivered commit is the revision containing this report. The final handoff provides its exact SHA.', '',
        '### Figures', '']
    for path in sorted((ROOT/'figures').glob('*.png')):lines.append(f'- [{path.stem}](figures/{path.name})')
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(verdict)


if __name__=='__main__':main()
