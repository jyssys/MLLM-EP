"""Write the bounded comparison from recorded serial measurements only."""
import json
from .common import *
from .priority_width_analysis import analyze_available


def number(value,digits=3):
    return 'NOT_MEASURED' if value is None else f'{value:.{digits}f}'


def median(value,key):
    return (value.get(key) or {}).get('p50')


def main():
    rows=analyze_available();plan=json.loads((ROOT/'state/priority_width_comparison_plan.json').read_text())
    assert all(r['measurements'][a].get('passed') for r in rows for a in ('loop','token')),'Do not publish an incomplete table as a complete comparison'
    qualifications=[]
    for name in plan['jobs']:
        if not name.endswith('_state8'):continue
        value=json.loads((ROOT/'raw'/name/'qualification_with_frozen_reference.json').read_text())
        assert value['campaign_gate_passed'],name
        qualifications.append(dict(name=name,passed=True))
    lines=['# Ouro and Huginn: matched Loop and Token-width comparison','',
        '## Result and scope','',
        'This report answers the requested short serial comparison. It does not report a new online Serving improvement.',
        'Each arm uses the same fixed BF16 model, recurrent depth, tuned projection policy, and Native connector.',
        'Every new split passes an N8 full-state qualification before its serial measurement.',
        'The state checks compare output tokens, valid KV, Hidden State, full recurrent work, handoff, ACK lifetime, and normal worker exits.',
        '',f'Completed split qualifications: {len(qualifications)}, with eight requests per qualification.',
        'Each serial cell uses eight requests within one fresh process start.',
        'Eight requests are not eight independent starts. This short comparison does not provide independent-start confidence intervals.',
        'The arms reuse the same eight model-specific prompt IDs. Request counts do not represent independent texts.',
        '', '## Matched work','',
        '| Model | Loop continuation | Token continuation | Recurrent rows per request |',
        '| --- | --- | --- | --- |']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["length"]} tokens × {r["D_loops"]} Loops | {r["token_suffix"]} tokens × {r["depth"]} Loops | {r["recurrent_rows_per_request"]} |')
    lines += ['',
        'The paired arms match recurrent token rows and recurrent dense FLOPs.',
        'They do not match total Attention, Prelude, prompt Coda, state bytes, or P critical-path saving.',
        'Both arms can group all remaining Loops in one scheduler submission when the Native frontier permits it.',
        'The Token arm is not restricted to one-loop host submissions.',
        'Actual submission counts appear below. A model submission is not one non-preemptive GPU kernel.',
        '', '## Fresh executed recurrent-core and D Prefill costs','',
        'CUDA Events measure the following enclosures in milliseconds.',
        'The core columns cover the actual packed recurrent graph instance, including output clones and possible host submission gaps.',
        'The D Prefill columns also include the submitted Prelude and prompt Coda work.',
        'They exclude separate first-output generation and state transfer. They are not Service TTFT.',
        'These measurements are not exclusive kernel unions, physical idle, or concurrent Decode interference.',
        '', '| Model / Token width | Loop core p50 | Token core p50 | Token / Loop core | Loop D Prefill p50 | Token D Prefill p50 | Token / Loop D Prefill |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        a,b=r['measurements']['loop'],r['measurements']['token']
        lines.append(f'| {r["model"]} / {r["token_suffix"]} | {number(median(a,"core_graph_sum_ms"))} | {number(median(b,"core_graph_sum_ms"))} | {number(r.get("Token_over_Loop_core_ratio"))} | {number(median(a,"D_prompt_sum_ms"))} | {number(median(b,"D_prompt_sum_ms"))} | {number(r.get("Token_over_Loop_D_prompt_ratio"))} |')
    lines += ['', '## Submission, source saving, and transfer accounting','',
        'P saving is a paired difference against the same-policy serial Native-path Zero control.',
        'It is an event-enclosure difference, not exclusive P kernel saving or measured online queue improvement.',
        'Native also transfers KV. The entire split payload is not incremental communication cost.',
        'GB means 1,000,000,000 bytes.',
        '', '| Model / width | Arm | P saved p50 ms | D Prefill p95 ms | Largest D submission ms | D submissions/request | Payload GB/request |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        for arm,m in r['measurements'].items():
            lines.append(f'| {r["model"]} / {r["token_suffix"]} | {arm} | {number(median(m,"P_saved_event_enclosure_ms"))} | {number((m.get("D_prompt_sum_ms") or {}).get("p95"))} | {number(m.get("largest_D_submission_ms"))} | {number(m.get("D_submissions_per_request"))} | {number(m["payload_bytes"]/1e9)} |')
    lines += ['',
        'Huginn Token128 uses a new 1920-row P prefix. The frozen policy has no tuned entry for that P geometry.',
        'That prefix uses the qualified original operator fallback. Its negative P saving is not proof of an inherent Token limitation.',
        'The D-side 128/256/512-row geometries all use frozen operator choices.',
        'A stronger end-to-end Token128 baseline requires separate P1920 tuning and qualification.',
        'Inner core-event counts differ with Loop count. The following control checks the smallest measured advantage.']
    control_path=ROOT/'analysis/priority_outeronly_control_results.json'
    if control_path.exists():
        control=json.loads(control_path.read_text())
        lines += ['', '## Outer-event-only control at Token512','',
            'Fresh serial controls remove every inner core event. Both arms retain two outer events on their single D submission.',
            'Output parity, full work, Native lifetime, and normal worker exit pass in both controls.',
            '', '| Arm | D Prefill p50 ms | Earlier inner-marked / outer-only ratio |',
            '| --- | --- | --- |']
        for r in control['rows']:
            assert r['passed']
            lines.append(f'| {r["arm"]} | {number(r["D_prompt_sum_ms"]["p50"])} | {number(r["marked_to_outeronly_ratio"],4)} |')
        lines += ['',f'Token / Loop D execution ratio: {number(control["token_over_loop"])}.',
            'The difference remains after removal of unequal inner-event counts.',
            'These controls also use different process starts. Their ratios combine observer effects and ordinary run variation.',
            'They do not prove an online profiler-overhead bound or provide independent-start confidence intervals.']
    lines += ['', '## Published tuned-GEMM confirmation at the same widths','',
        'This table reanalyzes the published isolated projection timers. It does not contain fresh recurrent-core timers.',
        'The fixed start-1 policy remains fixed in starts 2 and 3.',
        'Structural multiplicities match the paired rows above. A same-row GEMM does not distinguish a longer suffix from compatible batched suffixes.',
        'Those Attention layouts and readiness conditions remain different.',
        'The search is finite. It does not prove a globally optimal GEMM lower bound.',
        '', '| Model / Token width | Token / Loop dense sum, three starts |',
        '| --- | --- |']
    for r in rows:
        lines.append(f'| {r["model"]} / {r["token_suffix"]} | '+', '.join(number(x) for x in r['fixed_policy_dense_token_over_loop_three_starts'])+' |')
    lines += ['', '## Attention and boundary differences','',
        'The following values are structural counts, not measured kernel time or hardware traffic.',
        '', '| Model / width | Loop causal pairs per core layer | Token causal pairs per core layer |',
        '| --- | --- | --- |']
    for r in rows:
        p=r['structural_attention_causal_pairs_per_core_layer']
        lines.append(f'| {r["model"]} / {r["token_suffix"]} | {p["loop"]} | {p["token"]} |')
    lines += ['',
        'A suffix attends to the already-computed prefix at every remaining recurrent depth.',
        'A whole-prompt Loop contains the triangular causal work across the prompt.',
        'Huginn Loop continuation also performs full-prompt final Coda work on D.',
        'Token continuation performs Prelude and final Coda work only for its suffix.',
        'The measured component fields retain these differences. Equal recurrent rows alone are not full-cost fairness.',
        '', '## Interpretation limits','',
        'Wider Token suffixes can reduce or remove the isolated GEMM disadvantage.',
        'A faster GEMM does not remove dependencies between nonlinear recurrent passes.',
        'Different batching, fusion, persistent execution, or future kernels can change the measured frontier.',
        'This report does not establish an advantage under every Decode-safe execution budget.',
        '',
        'It does not establish online TTFT, goodput, or Decode non-inferiority for these new widths.',
        'It does not establish exclusive Loop-specific novelty over an equivalent logical-layer frontier.',
        'Hardware HBM traffic and exclusive kernel attribution remain NOT_MEASURED.',
        '', '## Evidence','',
        f'Evidence namespace: `{ROOT.relative_to(REPO)}`.',
        'The priority plan, immutable configurations, source snapshots, exact-state receipts, and per-request timers remain in that namespace.',
        '[Measured results and structural accounting](ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111/analysis/priority_width_comparison_results.json)',
        '[Frozen priority plan](ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111/state/priority_width_comparison_plan.json)',
        '', 'The broader workload-discovery campaign resumes after this bounded comparison.',
        'Its original deadline and GPU restoration rules remain unchanged.','']
    path=REPO/'poc_flashvep/reports/ouro_huginn_loop_token_suffix_128_256_512_quick_comparison.md'
    path.write_text('\n'.join(lines))
    write(ROOT/'state/priority_width_report_complete.json',dict(report=str(path),SHA256=digest(path),qualifications=qualifications))
    print(path)


if __name__=='__main__':main()
