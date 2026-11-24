"""Preserve aligned decode-tail evidence and negative runtime controls."""
import csv
import json
import time

from .common import ROOT, preserve, write


NAMES=['native_main100_timeline_CPC_q2_v1_Esync',
       'native_main100_timeline_CPC_q2_v2_client_host',
       'native_main100_timeline_CPC_q2_v3_API1ms',
       'native_main100_timeline_CPC_q2_v4_client_heap']


def main():
    preserve()
    rows=[];examples=[];qualifications=[]
    for name in NAMES:
        qpath=ROOT/'analysis'/f'{name}_qualification.json'
        dpath=ROOT/'analysis'/f'{name}_delivery_analysis.json'
        if not (qpath.exists() and dpath.exists()):continue
        q=json.loads(qpath.read_text());d=json.loads(dpath.read_text())
        qualifications.append(dict(name=name,observer_gate=q['observer_gate_passed'],comparison=q['comparison']))
        for stage,v in q['stages'].items():
            rows.append(dict(name=name,stage=stage,completed=v['completed_count'],
                TTFT_p95_ms=v['TTFT_p95_ms'],ITL_p99_ms=v['ITL_p99_ms'],
                ITL_max_ms=v['ITL_max_ms'],TPOT_p95_ms=v['TPOT_p95_ms'],
                all_ITL_pass=v['gate_attainment']['ITL_PASS'],joint_pass=v['gate_attainment']['JOINT_PASS'],
                raw_output_tokens_per_second=v['raw_output_tokens_per_second'],
                exact_profile_streams=d['exact_engine_API_client_streams'],
                observer_gate=q['observer_gate_passed'],evidence='DIAGNOSTIC'))
        examples.extend(dict(name=name,**x) for x in sorted(d['violating_client_gaps'],
            key=lambda x:x['client_ITL_ms'],reverse=True)[:4])
    assert rows
    with (ROOT/'analysis/native_main100_host_delivery_controls.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/native_main100_host_delivery_controls.json',dict(time=time.time(),
        evidence='TRACE_ANALYSIS of DIAGNOSTIC online controls',rows=rows,examples=examples,
        observer_qualification=qualifications,
        decisions=dict(API_1ms='NO_EVIDENCE_OF_DECODE_GATE_RECOVERY',
            client_heap='NO_DECODE_GATE_RECOVERY_IN_ONE_CPC' if any('v4_client_heap' in r['name'] for r in rows) else 'PENDING',
            native_quality='PENDING',online_token_split='NOT_RUN')))
    lines=['# Native MLLM100 Decode Delivery Diagnosis','',
        'Evidence: TRACE_ANALYSIS of DIAGNOSTIC online controls.',
        'These runs use Main100 at QPS2. They replay the same 5269 output tokens per run.',
        'Equal output length does not guarantee equal answer text.',
        'Every completed profile aligns EngineCore, API and client token IDs exactly for 100/100 requests.',
        'This alignment does not prove equality to the colocated model.',
        'The table retains clean, observed and clean-after results separately. They are not independent server starts.',
        'Task quality remains pending. No row is qualified Primary performance.','',
        '## Completed control results','',
        '| Run suffix | Stage | TTFT p95 ms | ITL p99 ms | Max ITL ms | TPOT p95 ms/token | All-ITL PASS % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["name"].split("_q2_")[-1]} | {r["stage"]} | {r["TTFT_p95_ms"]:.3f} | '
            f'{r["ITL_p99_ms"]:.3f} | {r["ITL_max_ms"]:.3f} | {r["TPOT_p95_ms"]:.3f} | {100*r["all_ITL_pass"]:.2f} |')
    lines += ['', '## Aligned examples','',
        '| Run suffix | Request | Token index | Client ITL ms | Engine ITL ms | API ITL ms | API-to-client next-token delay ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in examples:
        lines.append(f'| {r["name"].split("_q2_")[-1]} | {r["request_id"]} | {r["index"]} | '
            f'{r["client_ITL_ms"]:.3f} | {r["engine_ITL_ms"]:.3f} | {r["API_ITL_ms"]:.3f} | '
            f'{r["next_token_API_to_client_ms"]:.3f} |')
    lines += ['', '## Controlled and refuted claims','',
        'The D API switch interval changed from 5ms to 1ms. E, P and inference configuration stayed unchanged.',
        'The API process recorded its actual 1ms interval. A same-geometry N1 control matched all 16 output token IDs.',
        'This control did not restore 95% all-ITL attainment. It is not a successful decode-protection fix.',
        'A 218.786ms client GC occurred in v2. No active decode stream overlapped it.',
        'That event cannot explain the v2 ITL violations. The planned client-heap control was initially not run.',
        'A later v3 GC lasted 233.657ms during an active stream. The matching client ITL was 233.932ms.',
        'The corresponding Engine ITL was 14.636ms. The API ITL was 15.587ms.',
        'This new event supports a separate client-GC intervention. It does not explain all delivery tails.',
        'Another v3 client ITL was 658.081ms. Its Engine ITL was 6.602ms and API ITL was 622.461ms.',
        'That event overlaps API renderer processing and a 611.808ms API heartbeat delay.',
        'The client heartbeat delay was 4.018ms. No measured client append or GC overlapped that event.',
        'The API renderer enclosure includes waits and worker-thread work. It is not a CPU-exclusive cost.',
        'These examples refute GPU decode slowdown as the sole cause of the client tails.',
        'They do not identify the exact API operation that blocks delivery.',
        'API GC, preprocessing and IPC serialization need separate spans before another causal claim.',
        'The client heap intervention changes no server or inference configuration. Its result remains separate.',
        'The v4 client-heap control completed all three 100-request stages.',
        'Its all-ITL attainment was 71%, 71% and 72%. It did not restore the 95% target.',
        'The observed maximum client GC was 29.994ms. The observed profile maximum client ITL was 59.675ms.',
        'These maxima come from one sequence. They do not establish an independent-start confidence interval.',
        'The absence of a rare API stall in that sequence is not a proven effect of client GC policy.',
        '## Observer qualification','']
    for q in qualifications:
        lines.append(f'- `{q["name"]}`: aggregate observer gate `{q["observer_gate"]}`.')
    lines += ['',
        'The gate checks profile-versus-clean difference and clean-to-clean spread. Each limit is 5%.',
        'A failed gate can indicate an unstable clean baseline. It does not automatically prove profiler overhead.',
        'These diagnostic observations cannot provide qualified GPU-exclusive costs or an online speedup claim.',
        'All client-visible violations remain in the metrics. No timestamp correction removes delivery delay.',
        'CSV and JSON: `mllm_epd_prefill_heavy_token_split_20261124/analysis/native_main100_host_delivery_controls.*`.',
        'The source archive preserves current Python files. Boot-hash comparisons identify unavailable older source bytes.','']
    (ROOT.parent/'mllm_epd_client_decode_tail_diagnostic.md').write_text('\n'.join(lines))
    preserve();print('HOST_DELIVERY_DIAGNOSTIC_REPORT',len(rows),'rows',len(examples),'examples')


if __name__=='__main__':main()
