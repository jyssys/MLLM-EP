"""Use captured failures as a red-capable, CPU-only numerical feedback loop."""
import argparse
from .common import *
from .evidence import rows


def evidence():
    cases=[('default_rate_change','pilot_qwen8_l8000_o200_cfull_f0p25_v1','pilot_qwen8_l8000_o200_cfull_f0p5_v1'),
        ('async_repeat','diagnostic_qwen8_async_pd_l8000_o200_first_v1','diagnostic_qwen8_async_pd_l8000_o200_repeat_v1')]
    result=[]
    for label,a,b in cases:
        x={r['sample_id']:r for r in rows(a)};y={r['sample_id']:r for r in rows(b)};failures=[]
        for i,r in x.items():
            other=y.get(i)
            if other is None:failures.append(dict(sample_id=i,reason='MISSING_CANDIDATE'));continue
            assert r['input_ids']==other['input_ids'] and r['seed']==other['seed']
            first=next((j for j,(u,v) in enumerate(zip(r['output_ids'],other['output_ids'])) if u!=v),None)
            if first is not None:
                failures.append(dict(sample_id=i,first_divergent_output_index=first,
                    reference_token=r['output_ids'][first],candidate_token=other['output_ids'][first],
                    common_teacher_forced_history=r['input_ids']+r['output_ids'][:first],
                    prompt_id=r['prompt_id'],sampling_seed=r['seed']))
        result.append(dict(label=label,reference=a,candidate=b,compared=len(x),exact_sequences=len(x)-len(failures),
            failures=failures,passed=bool(x) and len(x)==len(y) and not failures,
            evidence='TRACE_ANALYSIS_OF_MEASURED_GPU_OUTPUTS',scope='This catches the actual token symptom. It does not identify an internal divergent operation.'))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--assert-parity',action='store_true');a=p.parse_args()
    result=evidence();write(ROOT/'state/ar_numerical_debug_original_failures.json',result)
    for r in result:print(r['label'],r['exact_sequences'],r['compared'],'PASS' if r['passed'] else 'FAIL',flush=True)
    if a.assert_parity:assert all(r['passed'] for r in result),'Original Qwen output reproducibility failure remains reproducible in captured GPU evidence'


if __name__=='__main__':main()
