"""Preserve each hypothesis update as an append-only research decision."""
import json
import time
from .common import ROOT


def record(identifier,stage,hypothesis,competitor,experiment,prediction,observation,next_action,evidence):
    path=ROOT/'state/research_decisions.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(r.get('id')==identifier for r in rows):
        return
    value=dict(id=identifier,time=time.time(),stage=stage,hypothesis=hypothesis,
        competing_hypothesis=competitor,experiment=experiment,prediction=prediction,
        observation=observation,next_action=next_action,evidence=evidence)
    with path.open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def initial_updates():
    record('A0003','STARTUP_CONTRACT_RECOVERY',
        'Native Full36 요청도 초기 Extension 생성자에는 유효한 Placeholder Cut이 필요하다.',
        'Native 모델 계산 또는 출력 경로가 잘못됐다.',
        'Native 초기 Cut을 기존 유효값으로 유지한다. CPU 회귀 검사 후 새 v2 이름으로 N8/N32를 실행한다.',
        '모델 계산을 바꾸지 않아도 시작 검사가 통과한다.',
        'CPU 4개 검사가 통과했다. Native N8/N32 State Gate가 통과했다. v1 실패 기록은 보존했다.',
        '실제 요청의 Full36와 Prefix/Suffix 방문 검사를 유지한다.','CORRECTNESS_ONLY')
    record('A0004','REPLAY_VALIDATION',
        '유효 P/D Service Demand와 Cohort Guard가 주요 Queue 증폭을 설명할 수 있다.',
        '관측하지 못한 KV/Host 상태가 대부분의 TTFT 차이를 지배한다.',
        '기존 Discovery에서 비용을 추정한다. 별도 Confirmation 54개 Start를 Queue 상태 갱신 Replay로 예측한다.',
        'TTFT p95 오차10%, Token/s 오차5%, Ready→D 오차max(10ms,20%) 기준을 검사한다.',
        '48/54가 세 기준을 모두 통과했다. Batch2 Last8/12의 일부 대기시간 예측은 실패했다.',
        '새 Seed 검증 전 비용 모델을 동결한다. 실패한 Cell에서 Oracle을 검증된 Serving 이득으로 주장하지 않는다.','OFFLINE_REPLAY')
    record('B0001','LAST2_QUALIFICATION',
        '작은 Last2도 기존 Exact Partial Forward와 Token Prefix 계약을 그대로 사용할 수 있다.',
        '432/448 Token 경계 또는 Layer2 State Handoff에서 정합성이 깨진다.',
        'Layer2와 Token432/448을 N8 다음 N32에서 검증한다.',
        'Original Greedy Output, 전체 Prompt KV, Hidden, Layer/Query 방문 및 ACK/Lifetime이 통과해야 한다.',
        'Layer2와 Token2 모두 N8/N32 State Gate를 통과했다. N8/N32는 Aggregate Matched Rows 성능 실험이 아니다.',
        'N36/72의 정확한 GEMM Rows와 Batch1 비용을 측정한다.','CORRECTNESS_ONLY')


if __name__=='__main__':
    initial_updates()
