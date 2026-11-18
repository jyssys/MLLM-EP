"""Plot matched-batch discovery without pooling different cohort sizes."""
import hashlib
import json
import time
from .common import ROOT, REPO, digest, write


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    path = ROOT/'analysis/independent_start_statistics.json'
    payload = path.read_bytes()
    data = json.loads(payload)
    statistics_hash = hashlib.sha256(payload).hexdigest()
    starts = [r for r in data['starts'] if r['stage'].startswith('batch1_')]
    for row in starts:
        check = ROOT/'analysis'/(row['name']+'_matched_batch_gate.json')
        assert json.loads(check.read_text())['passed']
    selected = dict(epoch=time.time(), starts=starts,
        aggregate=[r for r in data['aggregate'] if r['stage'].startswith('batch1_')],
        comparisons=[r for r in data['comparisons'] if r['stage'].startswith('batch1_')],
        scope='Actual P1/D-preparation1/Decode1. Do not pool the older P8/admission8 or P8/admission1 starts. Discovery N72 and confirmation N144 stay separate. Finite-cohort rates are not sustainable capacity.')
    write(ROOT/'analysis/batch1_summary.json', selected)
    output = REPO/'poc_flashvep'
    manifest = []
    colors = {0:'black', 4:'#0072B2', 8:'#D55E00', 12:'#009E73'}
    arms = [('Native',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]
    for cap in (16,64):
        qps = [2,4,6,8,10] if cap==16 else [2,6,10]
        values = [r for r in starts if r['stage']=='batch1_discovery' and r['cap']==cap]
        if not values:
            continue
        fig, axes = plt.subplots(2,2,figsize=(13,8))
        for method, layers in arms:
            lookup = {r['QPS']:r for r in values if r['method']==method and r['layers']==layers}
            label = ('Native' if method=='Native' else
                     f'Layer Last{layers}' if method=='Layer' else
                     'Token ~'+str({4:889,8:1778,12:2667}[layers]))
            series = [lookup.get(q) for q in qps]
            getters = [lambda r:100*r['regimes']['Common_Balanced_OutputSweep']['Joint_attainment'],
                       lambda r:r['mean_TTFT_ms']/1000,
                       lambda r:r['mean_TBT_ms'], lambda r:r['output_tokens_s']]
            for ax, get in zip(axes.flat, getters):
                y = [get(r) if r else np.nan for r in series]
                ax.plot(qps,y,color=colors[layers],linestyle='--' if method=='Token' else '-',
                        marker='s' if method=='Native' else 'o',markersize=4,label=label)
        for ax, label in zip(axes.flat,['Joint SLO attainment (%)','Mean TTFT (s)',
                                      'Mean TBT (ms)','Generated output tokens/s']):
            ax.set_xlabel('Offered QPS (requests/s)')
            ax.set_ylabel(label)
            ax.set_xticks(range(2,11))
            ax.grid(alpha=.25)
            ax.set_xlim(1.8,10.2)
        axes[0,0].set_ylim(-2,102)
        axes[0,0].legend(fontsize=8,ncol=2)
        fig.suptitle(f'Qwen3-8B | 2 x H100 | Input8000 / Output{cap} | Poisson\n'
                     'Actual P1 / D-prefill1 / Decode1 | B8192/S1024',fontsize=13)
        fig.text(.5,.012,'MEASURED discovery: N72, one independent start per point. '
                 'Only markers are measurements. Pending points are omitted. '
                 'Lines guide the eye. Finite-cohort drain rates.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.92))
        stem = output/f'dense_ar_batch_matched_four_metrics_o{cap}'
        for ext in ('png','pdf'):
            target = stem.with_suffix('.'+ext)
            fig.savefig(target,dpi=180)
            manifest.append(dict(path=str(target.relative_to(REPO)),SHA256=digest(target),
                                 output=cap,stage='batch1_discovery',count=72))
        plt.close(fig)
    write(ROOT/'analysis/batch1_figure_manifest.json',dict(epoch=time.time(),files=manifest,
        design_SHA256=digest(ROOT/'state/batch_matched_design_20261118.json'),
        statistics_SHA256=statistics_hash,source_starts=[r['name'] for r in starts],
        common_SLO=dict(TTFT_ms=5000,ITL_ms=34,TPOT_ms=33),
        scope='No original cap8 curve, interpolated measured point or N144 confirmation mean is substituted into the N72 discovery curve. Independent-start confirmation is reported separately.'))
    print('BATCH1_FIGURES',len(manifest),len(starts),flush=True)


if __name__ == '__main__':
    main()
