# H200 Slice-aware Embedding Transfer

D CUDA IPC consumer는 확정된 suffix에 필요한 원본 visual embedding row만 복제한다. Late-bound 결정 전에 P가 전체 이미지를 이미 복제했을 수 있으므로 절감량은 D clone에서 계산한다.

## 구현

1. E는 원본 BF16 embedding을 보유하고 기존 CUDA IPC descriptor를 발행한다. 두 transport arm은 같은 inline descriptor와 one-shot redirect 경로를 쓴다.
2. D가 import된 prefix 뒤의 embedding을 스케줄할 때 EC metadata에 필요한 원본 embedding row 범위를 담는다. 범위 계산에 vLLM의 placeholder-to-row mapping을 사용하므로 sparse embedding mask도 그대로 적용된다. 요청의 전체 prompt 길이와 NIXL prefix token 수를 확인한다.
3. D가 기존 CUDA IPC storage alias를 가져온 뒤 필요한 row slice를 D 소유 메모리에 clone한다. Cache entry는 V1 gather가 요청하는 원본 row 인덱스를 compact tensor 인덱스로 변환한다. Resident 범위를 벗어난 요청은 오류로 처리한다. Gather의 반환값은 기존 modality tag와 원본 token 위치를 그대로 받는 일반 tensor다.
4. D는 기존 clone-completion CUDA event를 기록하고 완료를 기다린 뒤 ACK를 보낸다. E는 발행한 descriptor의 ACK와 요청 완료를 확인한 뒤 source를 해제한다.

D worker에 `MLLM_EPD_SLICE_AWARE_EC=1`을 설정한 split 경로에서 동작한다. 같은 runtime에서 `campaign_slice_aware_ec_v1=false`를 보내 whole-image 대조군을 선택할 수 있다. 지원 조건은 vLLM V1, request-scoped media identifier, prefix cache off, non-speculative decode, KV import 실패 시 fail policy다. [CUDA IPC connector](../mllm_epd_prefill_heavy_token_split/cuda_ipc_ec.py) · [Embedding row mapping과 cache entry](../mllm_epd_h200_dynamic_correctness_slice_aware/embedding_rows.py)

Consumer-side 구현은 E의 원본 buffer와 descriptor를 재사용한다. Producer-side slice materialization은 late cut 이후 E allocation 또는 descriptor를 새로 만들어야 하므로 이번 구현에서는 사용하지 않았다. Crossing image의 D allocation과 복제 payload가 줄어들고 E의 원본 backing allocation과 이미지 단위 descriptor는 유지된다.

## 검증

CPU 회귀 7개가 통과했다. 7,823-token 단일 이미지의 한 row cut, sparse mask, 여러 후속 gather, 범위 위반, NIXL flag 변경, full/slice metadata 충돌, import 미확정 상태와 입력을 반복 제출할 때의 identifier 분리를 검사했다. [CPU 회귀](../mllm_epd_h200_dynamic_correctness_slice_aware/test_embedding_rows.py)

vLLM의 encoder cache key에는 `mm_feature.identifier`가 쓰인다. 연구 client는 request nonce와 frozen request ID, image index로 UUID를 생성한다. 같은 frozen row를 반복하는 부하에는 occurrence별 nonce를 쓴다. 관측된 120개 timing 요청에서 동일 이미지가 다른 요청에 들어갈 때에도 실제 media key는 달랐다. [Request UUID와 반복 처리](../mllm_epd_prefill_heavy_token_split/client.py)

실패와 client disconnect 시 기존 proxy는 D admission을 닫는다. E registry release는 성공으로 완료된 stream에만 호출한다. Slice entry는 D 소유 tensor와 기존 clone ACK 보호를 사용한다. 실패 경로 E source의 조기 해제는 방지되며, 중단된 요청의 source가 epoch 종료까지 남는 기존 cleanup 한계는 소스에서 확인했다. 별도의 cancellation/timeout GPU 회귀는 수행하지 않았다. [기존 proxy release 경로](../mllm_epd_prefill_heavy_token_split/proxy.py)

H200 정합성 실험은 같은 cut에서 whole과 slice를 각각 실행하고 imported/final Full28 KV, 원본 prompt ID 및 MRoPE, D query embedding signature, decode geometry, Natural EOS greedy token을 비교한다. Transfer 시간 측정은 별도의 probe-free 반복에서 실시한다. [동일 cut GPU 실험](mllm_epd_h200_same_cut_embedding_transfer_ablation.md)

## Crossing image descriptor 사전 발행

기존 inline redirect는 P가 이미 사용한 crossing image에서 E export RPC로 돌아갔다. `--preissue-crossing-ec`는 실제 image span 내부에 설정된 suffix 범위의 cut이 존재할 때 E에서 별도의 D용 descriptor를 발행한다. D는 기존 P inline claim을 먼저 시도한다. P가 이미 claim한 경우 D 전용 crossing descriptor를 시도한다. 사용 가능한 inline descriptor가 없으면 기존 RPC 경로가 유지된다. `--require-ready-transport`는 요청을 split하기 전에 모든 suffix image에 inline 경로가 있는지 검사한다. [Descriptor 선택 및 범위](../mllm_epd_prefill_heavy_token_split/crossing_ipc.py) · [옵트인 실행 설정](../mllm_epd_h200_modality_aware_dynamic_split/online.py)

두 descriptor는 같은 E 원본 tensor에서 각각 새 CUDA IPC ref-counter slot을 얻는다. 한 descriptor에는 D claim과 E revoke가 공유하는 단일 exclusive marker가 있다. D가 claim하면 clone 완료 ACK와 기존 source lifetime이 적용된다. E가 먼저 revoke에 성공하면 pinned Torch가 중복 IPC import에 쓰는 decrement를 해당 descriptor slot에 한 번 적용한다. 요청이 취소되면 같은 request key의 추가 발행을 막고 이미 발행된 미사용 descriptor만 회수한다. Claim된 descriptor와 원본 E buffer는 기존 소유권 계약에 남는다. [Torch pinned ref-counter 구현](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/csrc/StorageSharing.cpp#L291-L379) · [E Registry와 D ACK](../mllm_epd_prefill_heavy_token_split/cuda_ipc_ec.py)

IPC CPU 회귀 20개가 통과했다. Claim/revoke 동시 경합, claim 이후 ACK 전 source 유지, 취소 후 미사용 회수와 새 발행 차단, 단일 ref-counter slot decrement를 포함한다. [CPU 소유권 회귀](../mllm_epd_h200_dynamic_correctness_slice_aware/test_crossing_ipc.py)

Eager BI0 H200의 staged Natural10/20에서는 실제 D suffix를 4회 실행했다. Natural10의 N1/N8은 각각 D 전용 crossing descriptor 한 개와 suffix-only P redirect 다섯 개를 사용했다. 616-row 원본에서 필요한 458 rows를 복제했다. Natural20은 edge cut에서 각 실행마다 P redirect 한 개를 사용했다. 네 실행 모두 E export RPC 없이 clone ACK를 완료했다. E descriptor 199개 중 2개가 D ACK로, 197개가 미사용 revoke로 닫혔다. 465개 E source가 모두 해제됐고 최종 live source와 outstanding export는 0이었다. [Eager GPU 검증](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/preissue_eager_correctness.json)

같은 Eager 실행에서 두 N1 정합성은 통과했고 두 N8은 strict gate에 실패했다. Natural10은 8/8 Greedy가 일치했으며 Native-only follower 한 건의 P chunk `[15,4664,353]` 대 `[2318,2714]`에서 KV tolerance 차이가 D import 전에 나타났다. Natural20은 8/8 paired state tolerance를 통과했지만 실제 split lead와 Native-only follower 한 건의 Greedy가 달라졌다. Transport 및 source closeout 증거와 함께 이 실패 결과를 보존했다.

대칭 BI1 H200에서 같은 preissue 및 readiness guard로 Natural10과 Natural20의 N1 및 Mixed N8 strict gate를 모두 통과했다. 두 N8 모두 paired state 및 Greedy가 8/8 일치했다. Actual split lead는 Native follower와 최대 Batch8에서 각각 49/52 Decode step을 공유했다. 거부된 split의 same-request P continuation도 strict 통과했다. D suffix 4회에서 crossing spare 2개와 P redirect 12개를 사용했고 E export RPC는 0건이었다. 전체 E descriptor 183개는 D ACK 2건과 미사용 revoke 181건으로 닫혔다. E source 420개를 모두 해제했고 live source와 outstanding export는 0이었다. [BI1 GPU 정합성 및 closeout](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/preissue_bi1_correctness.json)
