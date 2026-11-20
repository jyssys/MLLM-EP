"""Test whether the N32 one-thread observation reproduces on full200."""
import json
import numpy as np
from .common import ROOT,write
from .confirmation_analysis import paired_ratio


def main():
    pairs=[]
    for seed in (2026112051,2026112052,2026112053):
        names={4:f'Main_Native_ShortAnswer200_q2.35_s{seed}_BI0_v3',
               1:f'Diagnostic_Native_T1_ShortAnswer200_q2.35_s{seed}_BI0_v3'}
        if not all((ROOT/'analysis'/f'{n}_metrics.json').exists() for n in names.values()):continue
        cfg={k:json.loads((ROOT/'raw'/n/'configuration.json').read_text()) for k,n in names.items()}
        assert cfg[1]['arrival_offsets']==cfg[4]['arrival_offsets']
        assert cfg[1]['dataset_SHA256']==cfg[4]['dataset_SHA256']
        values={k:json.loads((ROOT/'analysis'/f'{n}_metrics.json').read_text()) for k,n in names.items()}
        quality={k:json.loads((ROOT/'analysis'/f'{n}_quality_gate.json').read_text())['CI_passed'] for k,n in names.items()}
        pairs.append(dict(seed=seed,names=names,metrics=values,full200_quality=quality))
    fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms',
        'generated_output_tokens_per_second','SLO_qualified_output_tokens_per_second')
    result=dict(n=len(pairs),pairs=pairs,
        means={str(k):{f:float(np.mean([r['metrics'][k][f] for r in pairs])) for f in fields} for k in (1,4)},
        T1_over_T4={f:paired_ratio([r['metrics'][1][f] for r in pairs],[r['metrics'][4][f] for r in pairs]) for f in fields},
        all_quality_CI_passed=all(all(r['full200_quality'].values()) for r in pairs),
        evidence='MEASURED_ONLINE',
        scope='Thread-count intervention only. Same BI0 Natural200 trace, official UCX initialization and KV16. '
        'N32 exploratory observation and full200 confirmation remain separate. No posthoc replacement of the frozen default4-thread Main baseline.')
    write(ROOT/'analysis/BI0_NIXL_thread_full200_confirmation_v1.json',result)
    return result


if __name__=='__main__':main()
