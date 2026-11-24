"""Qualified low-load event brackets, without physical-idle or safety claims."""
import csv
import json

import numpy as np

from .common import ROOT,digest,preserve,write

NAME='ec_GPU_ENCLOSURES_C20_CPC_q0.5_v0'


def save(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    preserve()
    source=ROOT/'analysis'/f'{NAME}_enclosure_analysis_v0.json'
    value=json.loads(source.read_text());assert value['observer_gate_passed']
    central=value['ROIs']['central_arrival_window'];thresholds=[]
    for role in ['encoder','decode']:
        for row in central['stage_gap_statistics'][role]['threshold_coverage']:
            thresholds.append(dict(role=role,**row,evidence='Outside execute brackets, not physical idle'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for role in ['encoder','decode']:
        rows=[r for r in thresholds if r['role']==role]
        for ax,key,title in zip(axes,['count_fraction','duration_fraction'],
                               ['Window-count fraction','Duration-weighted fraction']):
            ax.plot([r['threshold_ms'] for r in rows],[100*r[key] for r in rows],'o-',label=role)
            ax.set_xscale('log');ax.set_xlabel('Minimum outside-bracket duration (ms)')
            ax.set_ylabel(title+' (%)');ax.set_ylim(0,101);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('QPS0.5 Main100 | CUDA-event observer PASS | Not decode-safe windows',fontsize=11)
    fig.text(.01,.01,'Unrelated streams and transfer can remain active. No additional prefill or serving gain is measured.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94])
    stem=ROOT/'figures/low_load_C20_outside_bracket_threshold_coverage_v0'
    for suffix in ['png','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=170)
    save(stem.with_suffix('.csv'),thresholds);plt.close(fig)
    run=NAME+'_profile';done=json.loads((ROOT/'state'/f'{run}_complete.json').read_text())
    left=done['monotonic_start'];right=left+20
    timeline=[]
    fig,ax=plt.subplots(figsize=(10,3.6))
    for index,role in enumerate(['encoder','prefill','decode']):
        folder=ROOT/'raw/gpu_enclosures'/NAME/role
        with (folder/'forward_enclosures.csv').open() as stream:rows=list(csv.DictReader(stream))
        pieces=[]
        for row in rows:
            lo=max(left,float(row['begin_ns'])/1e9);hi=min(right,float(row['end_ns'])/1e9)
            if hi<=lo:continue
            pieces.append((lo-left,hi-lo));timeline.append(dict(role=role,begin_relative_s=lo-left,
                end_relative_s=hi-left,meaning='Execute bracket, not exclusive kernel activity'))
        ax.broken_barh(pieces,(index-.25,.5),facecolors=f'C{index}')
    ax.set_yticks([0,1,2],['E','P','D']);ax.set_xlim(0,20);ax.set_xlabel('Seconds after first scheduled arrival')
    ax.set_title('First 20 seconds, selected before inspection | QPS0.5 event brackets',fontsize=10)
    ax.grid(axis='x',alpha=.2)
    fig.text(.01,.01,'Blank regions are outside the observed execute brackets. They are not certified GPU idle or safe slack.',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1])
    stem=ROOT/'figures/low_load_C20_aligned_execute_timeline_v0'
    for suffix in ['png','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=170)
    save(stem.with_suffix('.csv'),timeline);plt.close(fig)
    report=ROOT.parent/'mllm_epd_low_load_event_bracket_validation_v1.md';assert not report.exists()
    report.write_text('\n'.join([
        '# Low-load CUDA Event 계측 검증 — C20','',
        'QPS0.5의 Clean/Profile/Clean은 고정한 5% 계측 Gate를 통과했다. 독립 초기화는 1회다.',
        '이 판정은 실행 구간 계측에만 적용한다. Kernel-exclusive Busy 또는 Safe Headroom의 PASS가 아니다.',
        'Main100 Equal-output Replay를 사용했다. 각 실행은 원본 요청 100개와 Output5269 Tokens다.',
        'GPU4/5/6은 E/P/D다. Original BF16, BI0, 원본 Compiled 모델과 동일한 Arrival을 유지했다.',
        '모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.','',
        '## 관측 구간','',
        'Central Arrival Window는 예정 Arrival p10–p90 구간이다. 길이는 157.172988초다.',
        '이 구간은 Low Load다. Prefill-heavy 또는 Sustainable Steady State라고 부르지 않는다.','',
        '| Stage | Execute 구간 비율 | 구간 밖 비율 | 관측 Waiting 평균 | 관측 Waiting 최대 |',
        '|---|---:|---:|---:|---:|',
        '| E | 13.7370% | 86.2630% | 0.0000 | 0 |',
        '| P | 22.2888% | 77.7112% | 0.0520 | 3 |',
        '| D | 15.2372% | 84.7628% | 0.0000 | 0 |','',
        'Waiting은 Scheduler Sample의 값이다. 정확한 시간 가중 Queue는 아니다.',
        'E/D가 동시에 Execute 구간 밖인 시간은 Central Window의 72.3765%다.',
        '그 시간과 P Execute 구간의 겹침은 Central Window의 10.6240%다.',
        '같은 겹침은 P Execute 구간 시간의 47.6652%다. 두 분모를 합치지 않는다.',
        '이 비율은 Physical Idle Overlap이나 Actionable Slack이 아니다.','',
        'E의 구간 밖 Window는 Count 기준 p50 813.062ms, p95 4698.545ms다.',
        'D는 p50 0.023438ms, p95 1.625787ms다. D의 p99는 1201.151ms다.',
        'D의 짧은 Gap 수가 많다는 사실과 긴 Gap이 차지하는 시간 비율을 구분한다.',
        '0.1/1/5/10/20/50/100ms의 Count 및 Duration 가중 Coverage는 대응 CSV에 있다.','',
        '## 한계와 판정','',
        'Event는 원래 model-runner Execute를 둘러싼다. 그 안에는 Host Gap과 Stream Dependency가 있을 수 있다.',
        '구간 밖에서 별도 Stream의 Copy나 Export가 실행될 수 있다.',
        'Endpoint Clock Bracket의 최대 폭은 E252.344µs/P191.359µs/D163.526µs다.',
        '매우 짧은 Cross-GPU Gap의 정밀한 선후관계를 이 수치로 확정하지 않는다.',
        '추가 Prefill은 삽입하지 않았다. E/D-safe Quantum과 Safe Headroom은 미측정이다.',
        'QPS2의 C19 실패와 QPS4의 CUPTI 실패도 그대로 유지한다.',
        'Low-load 통과를 High-load 계측이나 Native 전체 Task Quality의 통과로 확대하지 않는다.','',
        '원시 Event, Clock Anchor와 Scheduler Sample은 `raw/gpu_enclosures/'+NAME+'/`에 있다.',
        '재계산 JSON과 CSV는 `analysis/'+NAME+'_enclosure_analysis_v0.json` 및 대응 CSV다.',
        'Threshold Figure와 CSV는 `figures/low_load_C20_outside_bracket_threshold_coverage_v0.{png,pdf,csv}`다.',
        'Timeline은 `figures/low_load_C20_aligned_execute_timeline_v0.{png,pdf,csv}`다.',
        '경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.','']))
    write(ROOT/'analysis/low_load_C20_report_provenance_v0.json',dict(source_SHA256=digest(source),
        observer_gate_passed=True,extra_prefill='NOT_RUN',safe_headroom='NOT_MEASURED',
        report_SHA256=digest(report),selection='First20s without inspection-based relocation'))
    preserve();print('LOW_LOAD_EVENT_REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
