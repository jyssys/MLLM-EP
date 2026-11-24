# E/P/D Idle Figure: Source와 관측 범위 감사

증거는 실제 Kernel·Copy Timeline의 TRACE_ANALYSIS다. 실패한 계측은 DIAGNOSTIC으로 표시했다.
같은 Manifest·Output Replay를 사용해도 Runtime Source와 관측 구간은 다를 수 있다.
낮은 Busy를 Decode-safe 또는 Encode-safe Window로 해석하지 않았다.

C25 QPS0.5와 C27 QPS3의 명시적 Engine Flag와 세 핵심 Source Hash는 같다.
비교한 Source는 proxy.py, processor_handoff.py, cuda_ipc_ec.py다. 전체 Software 동등성을 증명한 것은 아니다.
C11 QPS4는 이 Hash 조합이 다르다. 별도 Panel과 Fingerprint로 구분했다.
C25는 예정 Arrival p10–p90 전체를 관측했다. C27은 실제 Short ROI와 그 구간의 교집합만 관측했다.
C27과 C11은 고정한 5% Observer Gate를 통과하지 않았다. 두 점은 Primary GPU 시간으로 사용하지 않는다.
QPS 곡선은 연결하지 않았다. 측정하지 않은 QPS의 값은 보간하지 않았다.
Scheduler Waiting은 Step 관측별 평균이다. 시간 가중 Queue 또는 GPU-exclusive 대기가 아니다.

Aligned Figure는 적격 C25 중앙 구간의 Midpoint 전후 2초다. 결과를 보고 유리한 구간을 고르지 않았다.
GPU 간 매우 짧은 선후관계에는 Clock Anchor 오차가 있다. 대응 CSV에 Anchor 폭을 보존했다.
붉은 구간은 Copy 또는 Memset이다. 이를 실제 Network DMA 또는 Prefill 가능 시간으로 대체하지 않았다.
Main400은 그대로 보존했다. 이 Figure는 후속 Main100 실험이다.
SM Active·Tensor Core Activity·HBM Bandwidth는 미측정이다. Kernel Busy와 동일하다고 쓰지 않았다.

Figure와 CSV는 figures/idle_scope_*_v0에 있다. 모든 원시 Profile은 기존 Namespace에 남아 있다.
Config·원시 집계·Qualification SHA256과 제한된 Fingerprint 입력은 analysis/idle_scope_figure_audit_v0.json에 있다.
