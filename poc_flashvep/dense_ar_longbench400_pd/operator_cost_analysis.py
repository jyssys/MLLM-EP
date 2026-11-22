"""Keep kernel attribution and clean/profile/clean qualification separate."""
import json
from .common import ROOT,write
from .metrics import summarize
from .profile_analysis import analyze

FIELDS=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms')


def main():
    rows=[]
    for bucket,cut in ((0,2),(0,4),(3,4)):
        for axis in ('Layer','Token'):
            stem=f'Cost_I{bucket}_{axis}{cut}_O8_'
            names=[stem+s+'_v2' for s in ('Clean1','P_Operators','D_Operators','Clean2')]
            if not all((ROOT/'raw'/n/'dispatcher_result.json').exists() and
                json.loads((ROOT/'raw'/n/'dispatcher_result.json').read_text())['complete'] for n in names):continue
            metrics=[summarize(n) for n in names];a,b=metrics[0],metrics[-1]
            clean_ratios={f:a[f]/b[f] for f in FIELDS}
            for index,role in ((1,'prefill'),(2,'decode')):
                profile=metrics[index];decomp=analyze(names[index],role)
                ratios={f:profile[f]/((a[f]+b[f])/2) for f in FIELDS}
                stable=all(abs(v-1)<=.05 for v in clean_ratios.values())
                interference=all(abs(v-1)<=.05 for v in ratios.values())
                attribution=decomp['mapped_kernel_fraction']==1
                passed=stable and interference and attribution
                rows.append(dict(name=names[index],input_bucket=bucket,cut=cut,axis=axis,role=role,
                    profile_over_mean_clean=ratios,clean1_over_clean2=clean_ratios,
                    clean_stability_pass=stable,observer_5percent_pass=interference,
                    complete_kernel_attribution=attribution,qualified_kernel_measurement=passed,
                    evidence='ISOLATED_KERNEL_MEASUREMENT' if passed else 'DIAGNOSTIC',
                    phase_summary=decomp['phase_summary'],operator_kernel_union_ms=decomp['operator_kernel_union_ms'],
                    actual_GEMM_MNK_available=any(op['actual_GEMM_MNK'] for op in decomp['actual_operators']),
                    SM_utilization='NOT_MEASURED',HBM_traffic='NOT_MEASURED',
                    scope='One clean/profile/clean triplet. All16 requests use actual serving APIs. '
                        'Serial host latency is not online TTFT. Per-operator GPU unions are not host enclosures. '
                        'A faster profile beyond5percent also fails the conservative observer gate. '
                        'Graph decode kernels remain aggregate when Python operator attribution is unavailable.'))
    result=dict(rows=rows,scope='Both passed and failed observer cases remain visible. '
        'Do not transfer a gate from one axis or profile role to another. '
        'LM-head hooks were added between some frozen profiles. Compare common projection categories separately.')
    write(ROOT/'analysis/operator_cost_observer_audit_v2.json',result)
    print([(r['name'],r['evidence']) for r in rows],flush=True)
    return result


if __name__=='__main__':main()
