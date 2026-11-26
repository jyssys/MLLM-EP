# H200 MLLM Disaggregation Discovery PoC

Qwen2.5-VL-7B-Instruct에서 Track B Visual Understanding–Reasoning을 먼저 평가한다. 다음으로 Track A Visual Memory–Generation을 독립적으로 실행한다. 기존 H200 Token Split의 HOLD 결과는 시작 근거로 사용한다. [이전 실험](mllm_epd_h200_dynamic_slice_aware_final_summary.md) · [2026-11-27 Notion 연구 결정](https://app.notion.com/p/4220ee8277e48153a020e374317777f8)

## 동결 계약

- Base commit `e5fdd1260179568da654ca2c83f37cd8f26c880e`. 실험은 새 Git worktree와 외부 raw run 경로를 사용한다.
- Model/Processor revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. BF16, 원본 Encoder/Processor, LLM 28층, 원본 MRoPE, Greedy, Prefix Cache Off, Image-only.
- Main100 100개, MMMU Validation 40개, MathVista Testmini 30개, MMReason Testmini 30개. 150/50 grouped split 및 대표 24개는 실제 후보 결과가 나오기 전에 동결한다. Native 평가와 모든 후보에 같은 원래 질문을 적용한다. [Dataset 계약](mllm_discovery200_dataset_profile.md)
- Clean Native는 GPU4/5/6, BI0 NativeStrong-Free, NIXL 및 CUDA IPC EPD다. P/D B8192/S1024, KV42GiB, Block128, Natural EOS Cap2048을 적용한다. Historical Client SLO는 TTFT 5초 이하, 모든 ITL 34ms 이하, TPOT 33ms 이하로 유지한다.
- Open-loop 부하 탐색은 다른 모델 결과를 사용하지 않고 고정 Discovery 일부에서 QPS 0.75/2.5/5를 측정한다. Run 선택 규칙은 `native_epd.py`에 고정한다. Finite cohort 첫 도착부터 Drain까지의 처리율을 보고한다.
- GPU마다 UUID를 기록한다. 실제 비교 중 해당 GPU의 확인된 Burner만 중단하고 실험 밖 GPU의 Burner는 계속 동작한다. 구간별로 같은 GPU와 Topology를 유지한다. 외부 Burner의 공유 자원 간섭은 별도 대조가 있을 때만 수치화한다.

## 원본 모델 경계

Vision Encoder 32층과 Patch Merger는 입력 이미지를 3,584차원 Embedding으로 변환한다. Processor가 이미지당 Grid와 Image Token 위치를 제공한다. 모델은 그 위치에 Vision Embedding을 삽입하고 Text/Special 위치의 원본 Embedding을 유지한다. MRoPE는 이미지에 temporal, height, width 위치를 부여한다.

모든 Decoder Layer는 같은 Query/Key/Value Projection, Causal Attention, Output Projection, Residual과 SiLU MLP를 각 Query Row에 적용한다. 28개 Query Head와 4개 KV Head의 dimension은 각각 128이다. KV Cache에는 기존 위치를 적용한 K와 V가 저장된다. 최종 Normalization 뒤 LM Head에서 첫 출력 및 이후 Greedy Token을 선택한다. Visual Understanding과 Reasoning은 개입으로 검증할 역할 가설이다. 실제 실행 후보의 경계는 Vision Merger/Embedding 삽입, 각 Decoder Layer의 Q·KV/MLP, Prefill 완료 시점에 둔다. [Checkpoint 및 구현 Fingerprint](../mllm_specific_disaggregation_discovery/results/model_audit.json)

## Track B의 관측과 채택 기준

원본 Decoder Attention의 유효 causal key만 합산하고 Query/Key modality와 접근 가능한 key 비율을 함께 보고한다. Layer-wise attention mass와 paired 수치 변화를 학습에 의존하지 않는 진단으로 저장한다.

7개 연속 4-layer 그룹에서 동일 제거량의 Visual/Text Attention·MLP residual 개입과 KV 접근 개입을 비교한다. 실측 토큰 Prefix로 Teacher-forced 점수를 평가하고 Natural Greedy 결과를 별도로 평가한다. Dense Kernel을 그대로 실행한 mask와 residual ablation은 `CAUSAL_INTERVENTION`이다.

실제 동일 Shape의 Dense 비용과 두 GPU의 Activation/KV 이동을 계측한다. Modality placement는 동일 cut에서 label-blind control과 비교하며, 순서를 보존한 일반 Shape 기반 후보와 실제 지연을 비교한다. 약 1%p 이내의 품질 Point Estimate와 최적화 대상 비용 15% 감소 후보가 확인될 때 Full28 구현을 검토한다. Exact 후보는 Reduction Order에 따른 오차와 Greedy를 별도 확인한다.

## Track A의 관측과 채택 기준

Discovery200의 Visual/Text/Special/Generated KV 행 수와 KV Block을 계산하거나 관측 범위별로 보고한다. 같은 GQA, MRoPE, Causal Attention 계산에서 두 GPU가 같은 Query를 두 KV 파티션에 적용한다. Local logsumexp와 가중값을 FP32로 결합하고 full attention에 대한 오차와 Greedy를 확인한다.

Local Attention, 균형적인 일반 KV 분할, 고정 Prefix Residency, Visual Residency가 대조군이다. 초기 KV 이동 절감과 매 Layer Query/Partial 전달 실측을 분리하고 28개 Layer의 비용을 포함한다. 기존 DCP와 Prefix 분리보다 modality의 추가 이득을 요구한다.

## Evidence와 종료

Checkpoint, 실제 입력 Shape, Raw Timing, Source Fingerprint, Figure 입력을 저장한다. 포착한 모델 경로와 실제 Online Serving의 결과에는 각자 Evidence Level을 부여한다. Approximate 후보는 Quality–Work 대조를 거치고 Exact 후보는 수치 및 Greedy 대조를 거친다. Online 채택 기준은 비교 대상 동일 GPU와 품질에서 주요 latency 또는 SLO-goodput 5% 이상 개선 후보이며, 독립 N3 또는 신뢰구간을 보존한다. 검증한 실행 구조에 한정해 Track별 결론을 기록한다.
