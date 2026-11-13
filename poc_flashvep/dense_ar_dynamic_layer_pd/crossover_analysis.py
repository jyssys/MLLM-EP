"""Keep performance trade-offs separate from causal kernel interference."""
from .common import *
from .analyze import metrics,compare


def main():
    selection=json.loads((ROOT/'state/load_refinement_selection.json').read_text())
    rows=[]
    for cap in (16,32,64,128,200):
        for rate in (*selection['rates'][:1],4.777100014491616,*selection['rates'][1:]):
            if rate==4.777100014491616:
                native=f'lean_native_O{cap}_N64_v1';coloc=selection['selected'][str(cap)]
            else:
                native=f'refine_O{cap}_N64_q{rate:.3f}_native_v1';coloc=f'refine_O{cap}_N64_q{rate:.3f}_coloc_v1'
            a=metrics(native);b=metrics(coloc)
            if not a.get('complete') or not b.get('complete'):
                rows.append(dict(cap=cap,offered_QPS=rate,status='NOT_RUN',Native=native,Colocated=coloc));continue
            gate=compare(native,coloc)
            ratios={k:b[k]/a[k] for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','E2E_p95_ms','completed_QPS','output_tokens_s')}
            candidate=ratios['TTFT_p95_ms']<1 and max(ratios['ITL_p99_ms'],ratios['TPOT_p95_ms'])>1.05
            rows.append(dict(cap=cap,offered_QPS=rate,Native=native,Colocated=coloc,
                status='MEASURED' if gate['passed'] else 'FAILED_OUTPUT_GATE',output_gate=gate,
                Native_metrics=a,Colocated_metrics=b,Colocated_over_Native=ratios,
                performance_tradeoff_candidate=bool(candidate and gate['passed']),
                causal_prefill_Decode_kernel_interference='NOT_IDENTIFIED',
                qualified_same_window_P_kernel_saturation='NOT_MEASURED',
                scope='One N64 arrival seed per discovery cell. Qualified cohort DP2 does not admit new Prefill while its existing Decode cohort is active. These ratios do not prove kernel overlap.'))
    write(ROOT/'analysis/crossover_discovery_accounting.json',dict(results=rows,
        formal_verdict='NO_CONFIRMED_PD_COLOCATED_CROSSOVER_ON_TESTED_GRID',
        scope='Performance trade-off candidates exist only where measured. Physical P saturation failed metrology qualification. Concurrent chunked baselines failed exact output parity. Do not infer causal crossover from the cohort control.'))
    print('CROSSOVER_DISCOVERY_CELLS',len(rows))


if __name__=='__main__':main()
