"""Report diagnostic idle evidence without promoting it to safe capacity."""
import csv
import json

from .common import ROOT,preserve


def main():
    preserve()
    name='native_main100_CPC_q1_v0'
    data=json.loads((ROOT/'analysis'/f'{name}_idle_analysis.json').read_text())
    cpc=json.loads((ROOT/'analysis'/f'{name}_qualification.json').read_text())
    roi=data['ROIs']['central_arrival_window']
    lines=['# E/P/D Idle Profile: MLLM100 Diagnostic', '',
           '이 결과는 `DIAGNOSTIC` Online 실행의 `TRACE_ANALYSIS`다.',
           'Model Quality와 Observer Gate가 아직 통과하지 않았다. Primary GPU 시간으로 사용하지 않는다.',
           'Native EPD는 Original BF16 Qwen2.5-VL-7B와 실제 Online Encode를 사용했다.',
           'GPU0/1/2는 각각 E/P/D다. GPU UUID와 Source Hash는 대응 Server Config에 있다.',
           'P/D B8192·S1024, KV42GiB, Block128을 사용했다. E의 B는 114688이다.',
           '현재 Main100 SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`이다.',
           'QPS는 1.0이다. Arrival Seed는 2026112402, Order Seed는 2026112404다.',
           '요청별 기존 Natural 출력 길이를 고정해서 재생했다. 이는 Natural EOS 성능 결과가 아니다.', '',
           '## 중앙 Arrival 구간', '',
           f'분모는 Arrival p10부터 p90까지 {roi["duration_seconds"]:.6f}초다.',
           '이 구간을 Sustainable Steady State로 확정하지 않는다. 전체 Cohort 결과도 JSON에 따로 있다.', '',
           '| Stage | Kernel Busy | Kernel Inactivity | No Pending HTTP + Kernel Idle | Pending HTTP + Kernel Idle |',
           '|---|---:|---:|---:|---:|']
    table=[]
    for role,v in roi['roles'].items():
        fields=['kernel_busy_fraction','kernel_idle_fraction','no_pending_HTTP_kernel_idle_fraction','pending_HTTP_kernel_idle_fraction']
        lines.append('| '+role+' | '+' | '.join(f'{100*v[k]:.3f}%' for k in fields)+' |')
        table.append(dict(QPS=1.0,ROI='central_arrival_window',role=role,**{k:v[k] for k in fields}))
    lines += ['', 'Kernel Busy는 Kernel 구간의 Union이다. SM Active·Tensor Core Activity·HBM 포화율이 아니다.',
              'GPU Copy·Memset 구간은 별도로 보존했다. 이를 Prefill 실행 가능 시간에 그대로 더하지 않는다.',
              'Pending HTTP는 전처리·Transfer·Scheduler Queue·GPU Service를 모두 포함한다.',
              '따라서 Pending Idle을 Host Wait나 Transfer Wait 중 하나로 단정하지 않는다.',
              '현재 중앙 구간의 P Busy는 80% 기준에 못 미친다. 이 QPS에서 Prefill-heavy는 미입증이다.', '',
              '## Observer 확인', '',
              '| Metric | Profile / Clean 평균 차이 | 두 Clean 사이 상대 차이 |', '|---|---:|---:|']
    for key,v in cpc['comparison'].items():
        lines.append(f'| {key} | {100*v["profile_relative_difference"]:.3f}% | {100*v["clean_relative_spread"]:.3f}% |')
    lines += ['', 'Profile과 두 Clean 평균의 차이는 각 Metric에서 5% 이내였다.',
              '그러나 Clean 사이의 TTFT p95·ITL p99·TPOT p95 차이가 5%를 넘었다.',
              '따라서 안정적인 Baseline을 확인하지 못했다. Observer Gate는 통과하지 않았다.',
              '이를 Profiler의 인과적 간섭이 5%를 넘었다는 확정으로 바꾸지 않는다.',
              'CUPTI Dropped·Invalid Record는 세 Stage 모두 0이다.',
              'SM Active·Tensor Core Activity·HBM Bandwidth는 `NOT_MEASURED`다.',
              'Encode-safe·Decode-safe 추가 Prefill 실행도 `NOT_RUN`이다.', '',
              f'Raw Profile: `raw/profiles/{name}/`',
              f'집계: `analysis/{name}_idle_analysis.json`',
              f'Window CSV: `analysis/{name}_idle_windows.csv`', '']
    (ROOT.parent/'mllm_epd_encoder_decoder_idle_profile.md').write_text('\n'.join(lines))
    with (ROOT/'analysis'/f'{name}_stage_idle_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    both=roi['E_D_simultaneous_kernel_idle_fraction']
    p=roi['E_D_simultaneous_idle_during_P_busy_fraction_total']
    conditional=roi['E_D_simultaneous_idle_fraction_of_P_busy']
    text=['# Simultaneous E/D Idle: MLLM100 Diagnostic', '',
          '실측 Kernel Timeline을 공통 Host Monotonic Clock으로 정렬했다.',
          '각 Worker의 CUPTI 시작·종료 Anchor로 Affine 변환을 계산했다.',
          '이 결과는 `TRACE_ANALYSIS`다. Observer Gate와 Model Quality는 아직 미확정이다.', '',
          '| 구간 | 비율 | 분모 |', '|---|---:|---|',
          f'| E와 D의 동시 Kernel Inactivity | {both*100:.3f}% | 중앙 Arrival 구간 전체 시간 |',
          f'| P Busy 중 E와 D의 동시 Kernel Inactivity | {p*100:.3f}% | 중앙 Arrival 구간 전체 시간 |',
          f'| P Busy 중 E와 D의 동시 Kernel Inactivity | {conditional*100:.3f}% | 실제 P Kernel Busy 시간 |', '',
          'P Busy와 P Queue는 다르다. 위 표는 P Queue가 지속되는 시간의 비율이 아니다.',
          'Scheduler Queue는 Step별 관측값이다. HTTP Pending을 실제 GPU Queue로 대체하지 않았다.',
          '동시 Inactivity는 같은 요청을 세 GPU에서 병렬화할 수 있다는 증거가 아니다.',
          '미래 Encode·Decode Arrival과 비선점 Kernel의 비용은 이 표에 반영되지 않았다.',
          '따라서 Actionable Slack과 Encode/Decode-safe Headroom은 `NOT_MEASURED`다.',
          'E/D가 동시에 쉬는 현상은 진단 Timeline에서 관측했다. 정식 Headroom GO 판정은 보류한다.', '',
          'Idle Window Count와 Duration 가중 Threshold Coverage는 JSON에 따로 보존했다.',
          '0.1·1·5·10·20·50·100ms 기준을 모두 계산했다.',
          'PNG/PDF의 제목은 실제 Admission 가능 구간이 아닌 Physical Kernel Gap이라고 표시한다.',
          'Co-residency와 추가 Prefill Intervention을 실행한 상태가 아니다.', '']
    (ROOT.parent/'mllm_epd_simultaneous_idle_analysis.md').write_text('\n'.join(text))
    preserve();print('DIAGNOSTIC_IDLE_REPORTS_WRITTEN')


if __name__=='__main__':main()
