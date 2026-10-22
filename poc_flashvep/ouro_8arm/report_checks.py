"""Check printed benchmark cells against their measured source tables."""
import csv
import re
from .contract import *


def rows(name):return list(csv.DictReader((ROOT/name).open()))


def section_table(text,title):
    section=text.split('## '+title,1)[1].split('\n## ',1)[0]
    values=[]
    for line in section.splitlines():
        if not line.startswith('|'):continue
        parts=[v.strip() for v in line.strip('|').split('|')]
        if parts[0] in ARMS or parts[0] in ['Native TP1','Native TP4']:values.append(parts)
    return values


def main():
    result={r['arm']:r for r in rows('reviewed_eight_arm_results.csv')}
    replay={r['arm']:r for r in rows('replay_summary.csv') if r['family']=='TRACE_LOCKED_DYNAMIC_TOKENS' and r['C']=='4'}
    extension={r['run'].removeprefix('extension286_'):r for r in rows('extension286_serving.csv')}
    text=(ROOT/'benchmark_summary.md').read_text();cells=0
    actual=section_table(text,'Eight-arm quality and actual task-serving table');assert len(actual)==8
    for v in actual:
        r=result[v[0]]
        expected=[str(r['GPUs']),f"{r['quality_GSM8K_correct']}/300",f"{r['quality_MATH_correct']}/100",
            f"{float(r['service_cohort_sum_s']):.3f}",f"{float(r['task_cohort_speedup_vs_mode_single']):.3f}x",
            f"{float(r['task_mean_E2E_speedup_vs_mode_single']):.3f}x"]
        assert v[1:7]==expected,(v,expected);cells+=len(expected)
    clean=section_table(text,'Identical-work clean replay');assert len(clean)==8
    for v in clean:
        r=replay[v[0]];keys=['median_cohort_makespan_s','mean_TTFT_s','mean_TPOT_s','speedup_cohort_vs_mode_Single','overall_cohort_speedup_vs_S4']
        expected=[f"{float(r[k])*(1000 if k in ['mean_TTFT_s','mean_TPOT_s'] else 1):.3f}"+('x' if 'speedup' in k else '') for k in keys]
        assert v[1:]==expected,(v,expected);cells+=len(expected)
    preparation=section_table(text,'Fully instrumented preparation control');assert len(preparation)==8
    for v in preparation:
        r=extension[v[0]];expected=[f"{float(r[k]):.3f}" for k in ['cohort_makespan_sum_s','preparation_inclusive_cohort_sum_s']]
        assert v[1:]==expected,(v,expected);cells+=len(expected)
    quality={(r['run'],r['dataset']):r for r in rows('accuracy_by_arm.csv')}
    for arm in ARMS:
        for domain,N in [('gsm8k',300),('math500',100)]:
            assert int(quality['quality400_'+arm,domain]['N'])==N
    observers=rows('observer_validity.csv');assert len(observers)==18
    for row in observers:
        assert row['output_exact']==row['N']
        if row['depth_trace_exact']:assert row['depth_trace_exact']==row['N']
        if row['computed_argmax_exact']:assert row['computed_argmax_exact']==row['N']
    assert all(int(r['negative_ready_intervals'])==0 for r in rows('stage_timing_summary.csv'))
    archive=read(ROOT/'profile_archive_manifest.json');assert len(archive['exports'])==18
    for r in archive['exports']:
        assert r['SQLite_sha256']==r['decompressed_sha256'] and r['integrity_check']=='ok'
        assert sha(ROOT/r['lossless_export'])==r['compressed_sha256']
    documents=['analysis.md','benchmark_summary.md','correctness.md','memory_communication.md','profiling_analysis.md',
        'eight_arm_matrix.md','quality_results.md','serving_summary.md','README.md','status.md']
    links=0
    for name in documents:
        for target in re.findall(r'\]\(([^)]+)\)',(ROOT/name).read_text()):
            if target.startswith(('https://','http://','#')):continue
            path=target.split('#',1)[0]
            assert (ROOT/path).resolve().exists(),(name,target)
            links+=1
    write(ROOT/'state/final_report_validation.json',dict(benchmark_numeric_cells_checked=cells,
        quality_arms=8,questions_per_arm=400,observer_comparisons=18,lossless_archives=18,local_links_checked=links,
        status='PASS',scope='Printed benchmark cells, artifact integrity, and local links. Not an additional numerical or population-quality qualification.'))
    print('REPORT_CHECKS_PASS',cells,'benchmark cells',links,'local links',flush=True)


if __name__=='__main__':main()
