"""Report completed Natural Main100 pilots without promoting their evidence."""
import csv
import json
import time

from .common import ROOT, digest, preserve, write
from .metrics import summarize


RUNS={
    'Agg DP3':[(.5,'agg_main100_C3_protocol_DIAGNOSTIC_q05_v0'),
               (2.,'agg_main100_capacity_DIAGNOSTIC_q2_v0')],
    'E + PD DP2':[(.5,'e_pd_main100_capacity_DIAGNOSTIC_q05_v0'),
                  (2.,'e_pd_main100_capacity_DIAGNOSTIC_q2_v0')],
    'Native EPD':[(.5,'native_main100_C3_protocol_DIAGNOSTIC_q05_v0'),
                  (2.,'native_main100_capacity_DIAGNOSTIC_q2_v0')],
}


def main():
    preserve()
    rows=[]
    for method,runs in RUNS.items():
        for qps,run in runs:
            value=summarize(run)
            config=json.loads((ROOT/'configs'/f'{run}.json').read_text())
            assert config['output_mode']=='Natural EOS'
            assert config['QPS']==qps
            assert value['planned_count']==value['completed_count']==100
            row=dict(method=method,QPS=qps,start_count=1,evidence='DIAGNOSTIC_C3',
                run=run,raw_SHA256=value['raw_SHA256'],
                config_SHA256=digest(ROOT/'configs'/f'{run}.json'),
                manifest_SHA256=config['workload']['manifest_SHA256'],
                duration_seconds=value['duration_seconds'],output_tokens=value['total_output_tokens'])
            for field in ['TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms',
                          'ITL_max_ms','TPOT_p95_ms','raw_output_tokens_per_second',
                          'completed_requests_per_second','total_input_output_tokens_per_second','SLO_goodput']:
                row[field]=value[field]
            row.update(value['gate_attainment'])
            rows.append(row)
    assert len({r['manifest_SHA256'] for r in rows})==1
    target=ROOT/'analysis/main100_baseline_motivation_pilot.csv'
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/main100_baseline_motivation_pilot.json',dict(time=time.time(),rows=rows,
        evidence='DIAGNOSTIC_C3',qualified_primary=False,quality_gate='PENDING',
        independent_start_CI='NOT_AVAILABLE: one server start per method/QPS',
        selection='All six completed original Natural Main100 pilot cells. Do not pool later fixed-output or runtime controls.',
        motivation_decision='NOT_YET_QUALIFIED',prefill_heavy_decision='NOT_YET_CONFIRMED'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=[('TTFT_p50_ms','TTFT p50 (ms)'),('TTFT_p95_ms','TTFT p95 (ms)'),
           ('ITL_mean_ms','Mean client TBT (ms)'),('ITL_p99_ms','Client ITL p99 (ms)'),
           ('TPOT_p95_ms','Request TPOT p95 (ms/token)'),
           ('total_input_output_tokens_per_second','Input + output tokens/s'),
           ('raw_output_tokens_per_second','Output tokens/s'),('SLO_goodput','SLO-qualified output tokens/s'),
           ('completed_requests_per_second','Completed requests/s'),
           ('TTFT_PASS','TTFT attainment'),('ITL_PASS','All-ITL attainment'),('JOINT_PASS','Joint attainment')]
    for field,label in plots:
        fig,ax=plt.subplots(figsize=(5.6,4))
        for method in RUNS:
            points=[r for r in rows if r['method']==method]
            ax.plot([r['QPS'] for r in points],[r[field] for r in points],marker='o',label=method)
        ax.set(xlabel='Offered QPS (requests/s)',ylabel=label,
               title='Main100 Natural EOS | DIAGNOSTIC')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.text(.5,.01,'N=1 per point. Quality pending. No start-level CI. Finite cohort.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.04,1,1])
        path=ROOT/'figures'/f'main100_motivation_{field}'
        for ext in ['png','pdf']:fig.savefig(str(path)+'.'+ext,dpi=170)
        plt.close(fig)
        with (ROOT/'analysis'/f'main100_motivation_{field}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['method','QPS','start_count',field,'run','evidence'])
            writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    lines=['# MLLM100 EPD Motivation Pilot','',
        'Evidence: DIAGNOSTIC_C3. Status: preliminary. Task quality remains pending.',
        'Each point uses one server start. These points do not provide a start-level confidence interval.',
        'The initial Natural EOS pilots use Main100. They do not use the later fixed-output runtime controls.',
        'Each method uses three H100 GPUs. All points use the same manifest, request order and arrival seed.',
        'Encoder work runs online. The EC handoff uses the installed debug connector and tmpfs files.',
        'This EC transport is not a production NIXL embedding connector. The P-to-D KV handoff uses NIXL.',
        'The installed source and its local changes must be distinguished from an upstream version label.','',
        '## Latency and throughput','',
        '| Method | QPS | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Output tok/s | Total tok/s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["QPS"]:g} | {r["TTFT_p50_ms"]:.3f} | {r["TTFT_p95_ms"]:.3f} | '
            f'{r["ITL_p99_ms"]:.3f} | {r["TPOT_p95_ms"]:.3f} | {r["raw_output_tokens_per_second"]:.3f} | '
            f'{r["total_input_output_tokens_per_second"]:.3f} |')
    lines += ['', '## SLO','',
        'Thresholds: TTFT ≤5000ms, every client ITL ≤34ms, request TPOT ≤33ms/token.',
        'Joint PASS requires all three conditions. Every attainment denominator is 100 requests.',
        'A low token-level ITL p99 does not prove that every ITL passes for each request.','',
        '| Method | QPS | TTFT PASS % | All-ITL PASS % | TPOT PASS % | Joint PASS % | SLO output tok/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["QPS"]:g} | {100*r["TTFT_PASS"]:.2f} | '
            f'{100*r["ITL_PASS"]:.2f} | {100*r["TPOT_PASS"]:.2f} | {100*r["JOINT_PASS"]:.2f} | {r["SLO_goodput"]:.3f} |')
    lines += ['', '## Interpretation and limits','',
        'At QPS2, Native EPD has lower ITL p99 than Agg DP3. It also has higher TTFT p95.',
        'This is an observed trade-off in an unqualified pilot. It is not a confirmed motivation result.',
        'Native EPD passes the all-ITL gate for 74/100 requests at QPS2. It fails the 95% absolute decode target.',
        'No method or request is removed because its result is unfavorable.',
        'Natural output counts differ between methods. This table does not control output work.',
        'Throughput uses planned first arrival through drain. It is finite-cohort throughput, not sustainable capacity.',
        'Total throughput counts each request input and output once. It does not count tokens again at each E/P/D stage.',
        'TTFT includes dispatch lateness and all stage waits. Client ITL includes delivery delay.',
        'TPOT is (last token time − first token time)/(output count − 1). Single-output TPOT is zero.',
        'Mean TBT averages all client inter-token intervals. Co-delivered tokens have zero intervals.',
        'Runtime host enclosures are not GPU-exclusive service times.',
        'E/P/D physical idle results remain diagnostic. No additional prefill work has established safe headroom.',
        'The main CSV records raw hashes, config hashes, counts and durations.',
        'CSV: `mllm_epd_prefill_heavy_token_split_20261124/analysis/main100_baseline_motivation_pilot.csv`.',
        'Figures: `mllm_epd_prefill_heavy_token_split_20261124/figures/main100_motivation_*.{png,pdf}`.',
        'Each figure has a matching CSV. No omitted QPS is interpolated or represented as a measured point.','']
    (ROOT.parent/'mllm_epd_duetserve_style_motivation.md').write_text('\n'.join(lines))
    preserve()
    print('DIAGNOSTIC_BASELINE_REPORT',len(rows),'cells',len(plots),'figures')


if __name__=='__main__':main()
