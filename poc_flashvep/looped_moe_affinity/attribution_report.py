"""Render measured attribution only. Keep queued work separate from results."""
from collections import defaultdict
from .common import ROOT

def choose_observer(rows):
    """Keep failed fine pilots separate from passing coarse pilots."""
    variants=defaultdict(list)
    for row in rows:
        key=(row['model'],row['ep'],row['batch'],row['prefix_tokens'],row['policy'],row['detail'],row['timing_status'],row.get('allocator_conf','DEFAULT'))
        variants[key].append(row)
    cells=defaultdict(list)
    for key,values in variants.items():
        total=sum(value['mean_component_pct'] for value in values)
        assert abs(total-100)<1e-5,(key,total)
        cells[key[:5]].append(values)
    selected=[]
    for alternatives in cells.values():
        best=min(alternatives,key=lambda group:(group[0]['timing_status']!='LOW_OVERHEAD_OBSERVER',
            abs(group[0]['observer_overhead_mean_ratio']-1),group[0]['detail']))
        selected.extend(best)
    return selected

def render(rows):
    selected=choose_observer(rows)
    lines=['# Whole-E2E component attribution','',
        '## Status','',
        'This file contains only completed attribution runs.',
        'The attribution campaign waits for the active accuracy and control campaigns.',
        'A queued configuration is not a measured result.','',
        '## Measurement contract','',
        'The denominator is the complete observed CUDA-event span for 32 greedy output tokens.',
        'The rollout includes prefill, subsequent forwards, sampling, and output completion.',
        'These official model paths recompute the full prefix. They do not use cached decode.',
        'EOS is disabled in this hardware comparison. Official benchmark accuracy uses separate task protocols.','',
        'Each component occupies an exclusive part of the endpoint timeline.',
        'Nested intervals do not count twice. Unrounded component percentages sum to 100%.',
        'The table does not sum different ranks or multiply one forward by the output length.','',
        'These API spans include GPU execution, submission gaps, and stream waits.',
        'A dispatch or combine span is not pure network transfer time.',
        'An expert span is not pure GEMM service time.',
        'Rank waiting remains inside the corresponding communication span unless a separate measurement can identify it.','',
        'Clean host latency and observer latency remain separate.',
        'The timing gate requires at most 5% mean observer overhead and at most 1% event-to-host disagreement.',
        'A passing gate does not prove unbiased clean component shares.',
        'The report never rescales observer percentages into claimed clean measurements.','',
        '## B1, prefix 128, original static EP','',
    ]
    if not selected:
        lines.extend(['Status: **PENDING_MEASUREMENT**.',
            'No whole-E2E component percentage is available yet.',
            'Existing per-forward traces do not replace this measurement.',''])
    overview=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [1,4,8]:
            values=[v for v in selected if (v['model'],v['ep'],v['batch'],v['prefix_tokens'],v['policy'])==(model,ep,1,128,'static')]
            if not values:continue
            first=values[0]
            overview.append(first)
            lines.extend([f'### {model}, native' if ep==1 else f'### {model}, EP{ep}','',
                f'Observed event span: {first["observed_e2e_gpu_span_mean_ms"]:.3f} ms.',
                f'Observed host E2E: {first["observed_host_e2e_mean_ms"]:.3f} ms.',
                f'Clean host E2E: {first["clean_host_e2e_mean_ms"]:.3f} ms.',
                f'Observer/clean latency ratio: {first["observer_overhead_mean_ratio"]:.4f}.',
                f'Timing status: `{first["timing_status"]}`. Detail: `{first["detail"]}`.',
                f'Allocator: `{first.get("allocator_conf","DEFAULT")}`.',
                f'Traced rollouts: {first["traced_rollouts"]}.','',
                '| Component | Mean ms | Share of observed E2E | 95% CI, percentage points |',
                '|---|---:|---:|---|'])
            for row in values:
                low,high=row['ci95_low_pct'],row['ci95_high_pct']
                interval=f'[{low:.3f}, {high:.3f}]' if isinstance(low,(int,float)) else 'Not available: one trace'
                lines.append(f'| {row["component"]} | {row["mean_component_ms"]:.3f} | {row["mean_component_pct"]:.3f}% | {interval} |')
            lines.extend(['| Total | '+f'{sum(v["mean_component_ms"] for v in values):.3f} | 100.000% | Not applicable |','',
                'The interval describes serial rollouts within one process start.',
                'It is not an independent-server-start confidence interval.',
                'A zero attributed interval does not prove absent arithmetic.',
                'A coarse observer can include that work in an enclosing region.',''])
    lines.extend(['## Complete data','',
        '- `e2e_component_percentages.csv` retains each observer detail and timing status.',
        '- `e2e_component_percentages_raw.csv` retains each traced rollout and its evidence path.',
        '- `e2e_component_phase_percentages.csv` separates first output completion from the remaining outputs.',
        '- `raw/*_e2e_share_*/attribution_status.json` retains clean repetitions and observer timing gates.',
        '- `e2e_attribution_contract.md` defines the accounting rules and interpretation limits.','',
        'The first-output phase includes prefill and first-token sampling.',
        'Phase component percentages identify both their phase denominator and the whole-E2E denominator.',
        'A lower component percentage alone does not prove a lower latency. Compare the raw milliseconds too.',''])
    (ROOT/'e2e_breakdown.md').write_text('\n'.join(lines))
    if overview:plot(selected)

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .aggregate_attribution import COMPONENTS
    cells=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [1,4,8]:
            group=[v for v in rows if (v['model'],v['ep'],v['batch'],v['prefix_tokens'],v['policy'])==(model,ep,1,128,'static')]
            if group:cells.append((model,ep,group))
    fig,ax=plt.subplots(figsize=(14,max(5,len(cells)*.65)))
    left=[0.]*len(cells)
    colors=plt.get_cmap('tab20')
    for index,component in enumerate(COMPONENTS):
        values=[next(v['mean_component_pct'] for v in group if v['component']==component) for _,_,group in cells]
        ax.barh(range(len(cells)),values,left=left,label=component,color=colors(index%20))
        left=[a+b for a,b in zip(left,values)]
    ax.set_yticks(range(len(cells)),[f'{model}, {"native" if ep==1 else "EP"+str(ep)} (n={group[0]["traced_rollouts"]})' for model,ep,group in cells])
    ax.set_xlim(0,100);ax.set_xlabel('Share of observed whole-E2E event span (%)')
    ax.set_title('LIVE observer: B1, prefix 128, 32 greedy tokens, full-prefix recomputation')
    ax.invert_yaxis();ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    fig.text(.01,.01,'Exclusive API spans, including launch gaps and waits. Clean latency is separate. See CSV timing gates.',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1));(ROOT/'figures').mkdir(exist_ok=True)
    for suffix in ['svg','png']:fig.savefig(ROOT/'figures'/f'e2e_component_percentages.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    from .aggregate_attribution import main
    main()
