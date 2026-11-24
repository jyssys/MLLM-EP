"""Render future parallelism requirements, not unimplemented performance."""
import time

from .common import ROOT,digest,preserve,write
from .token_discovery_analysis import table

ROWS=[
    dict(candidate='Request-level elastic',ready_state='Real encode complete\nWeights + KV capacity',
        communication='E worker: final KV to D\nD deflection: no P prefix KV',
        original_service_risk='Nonpreemptible prefill\nNew E/D arrivals can wait',
        current_evidence='D-only C3 control: NO_GO\nE/P/D elastic: NOT_RUN'),
    dict(candidate='Inter-request token pipeline',ready_state='Previous chunk KV\nCurrent embeddings + MRoPE',
        communication='Chunk KV + embedding handoff\nCredit + clone ACK',
        original_service_risk='Admission backpressure\nStage queue + buffer lifetime',
        current_evidence='P/D continuation implemented\n3-stage token pipeline: NOT_RUN'),
    dict(candidate='Layer-wise wavefront',ready_state='All earlier chunks: same-layer KV\nCurrent chunk: previous-layer hidden',
        communication='Layer-level KV ready exchange\nRepeated dependency barriers',
        original_service_risk='Short windows may not cover\nTransfer + dependent compute',
        current_evidence='Dependency accounting only\nActual wavefront: NOT_RUN'),
    dict(candidate='Elastic context parallelism',ready_state='Same-layer participant state\nWeights + KV + attention metadata',
        communication='Required attention KV exchange\nFinal decode-KV placement',
        original_service_risk='Participant entry/exit barriers\nEncode/decode deadline risk',
        current_evidence='Structural requirements only\nElastic CP: NOT_RUN'),
]


def main():
    preserve();name='parallelism_feasibility_structural_v0'
    report=ROOT.parent/'mllm_epd_dual_side_parallelism_feasibility_v1.md';assert report.is_file()
    assert not (ROOT/'figures'/f'{name}.csv').exists()
    records=[dict(**row,evidence='STRUCTURAL_ACCOUNTING',future_3GPU_speedup='NOT_MEASURED',
        source_report_SHA256=digest(report)) for row in ROWS]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(16,5));axis.axis('off')
    keys=['candidate','ready_state','communication','original_service_risk','current_evidence']
    grid=axis.table(cellText=[[row[key] for key in keys] for row in ROWS],
        colLabels=['Candidate','Required ready state','Communication','Original E/D service risk','Current implementation/evidence'],
        cellLoc='left',loc='center',colWidths=[.17,.23,.22,.19,.23])
    grid.auto_set_font_size(False);grid.set_fontsize(8.5);grid.scale(1,3.6)
    axis.set_title('STRUCTURAL_ACCOUNTING | No measured 3GPU speedup | E/D physical idle is not safe admission',pad=18)
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
    plt.close(fig);table(ROOT/'figures'/f'{name}.csv',records)
    write(ROOT/'analysis'/f'{name}_audit.json',dict(time=time.time(),source_report_SHA256=digest(report),
        evidence='STRUCTURAL_ACCOUNTING',actual_new_parallelism='NOT_RUN',claimed_speedup=False))
    preserve();print('FUTURE_PARALLELISM_STRUCTURAL_FIGURE_SAVED',flush=True)


if __name__=='__main__':main()
