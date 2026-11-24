# Exact multimodal token-prefix contract — CPU implementation checkpoint

상태: `CPU_CONTRACT_PASS`. GPU Token Split 정합성은 `NOT_RUN`이다.
이 문서는 Online 성능 결과가 아니다.

## 고정 조건

모델은 Qwen2.5-VL-7B-Instruct다. Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Original BF16과 BI0를 유지한다. LLM은 28 Layers다. Vision Encoder와 Processor는 수정하지 않는다.
실험용 Main100 SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`다.
Main400과 기존 결과는 보존한다.

## 실행 계약

전체 Prompt 길이는 L이다. P가 처리할 Prefix 길이는 k다. 조건은 `0 < k < L`이다.

1. E는 원래 이미지 전체를 Online으로 처리한다.
2. 공식 Renderer와 InputProcessor가 원래 전체 Prompt를 처리하고 검증한다.
3. P의 EngineCoreRequest만 `prompt_token_ids[:k]`로 제한한다.
4. 이미지 순서, 원래 Placeholder 범위, Image Grid와 Multimodal Feature는 그대로 보존한다.
5. P는 원래 Compiled Full28 모델을 사용한다. Model Forward와 Attention Forward는 교체하지 않는다.
6. P는 실제 Prefix KV를 NIXL로 제공한다. P API의 임시 Sample 하나는 버린다.
7. D는 원래 전체 Prompt와 전체 MRoPE 정보를 유지한다.
8. D는 실제 Prefix 길이 k만 Remote KV로 등록한다. 이후 원래 Scheduler가 Suffix를 처리한다.
9. 첫 Client Output은 Suffix 전체 처리 후에만 생성한다. 이후 Full28 Decode를 수행한다.

P의 임시 Sampling과 LM-head 비용은 추가 비용이다. 이 비용을 Prefill 절감으로 숨기지 않는다.
NIXL Block 전송은 Token 길이와 다를 수 있다. 유효 KV 범위와 전송 Byte 수를 따로 검증해야 한다.

## 이미지 내부 경계

원래 이미지 구간을 `[a,b)`로 둔다. P와 D의 실제 Query 범위는 각각 `[0,k)`와 `[k,L)`다.
공식 Multimodal Scheduler가 두 Query 범위에 필요한 Embedding 부분을 선택한다.
`a < k < b`이면 같은 이미지의 원래 전체 Embedding이 두 소비자에게 필요하다.
이미지를 다시 Encode하거나 Placeholder를 임의로 줄이지 않는다.

P Prefix의 MRoPE Delta를 D Decode에 사용하지 않는다. D는 원래 전체 Prompt의 Delta를 사용한다.
CPU 위치 감사는 원래 Qwen 위치 함수의 Prefix 일치를 확인했다. 실제 GPU 상태의 일치는 아직 미검증이다.

## CPU 검증

실행 명령:

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest poc_flashvep.mllm_epd_prefill_heavy_token_split.test_token_split
```

13개 테스트를 통과했다. 실제 EngineCoreRequest의 Msgpack 직렬화도 검사했다.
구현 전 Import 실패와 필수 Test Fixture 필드 누락 로그도 보존했다.

검사 범위:

- Native 요청은 변경하지 않는다.
- P의 Token ID만 제한한다. 원래 Multimodal Feature와 이미지 구간을 유지한다.
- D는 전체 Prompt를 유지한다.
- NIXL은 실제 Prefix 길이만 Match한다.
- Producer가 다른 길이를 보고하면 실패한다.
- 이미지 내부의 비정렬 경계를 조용히 다른 경계로 변경하지 않는다.

로그는 Campaign의 `logs/cpu_token_contract_wire_v3.log`다.
이 테스트는 GPU KV 정합성이나 Task Quality를 증명하지 않는다.

## GPU Gate와 미실행 항목

다음 검사는 아직 남아 있다.

- 이미지 직전·직후·내부, 이미지 사이, Text 구간의 실제 GPU 실행.
- N1과 N8의 Position, Prepared Embedding, Full28 Prefix KV 및 최종 Prompt KV 비교.
- GPU Query Coverage와 중복·누락 여부 검사.
- Natural EOS의 출력과 Task Quality 검사.
- 실제 Decode-safe Window와 추가 작업의 Decode 보호 검사.
- Strong Static Split과 Idle-aware Split의 C3 독립 반복.

초기 GPU 검증은 Active Decode가 없는 정합성 전용 실행으로 제한한다.
그 실행의 시간을 Serving Speedup으로 사용하지 않는다.
현재 Native EPD의 전체 Task Quality Gate는 미확정이다.
MMLongBench-Doc의 공식 GPT 답변 추출은 승인된 API가 없어 미실행이다.
이 한계를 Raw Answer 점수로 대체하지 않는다.

## 전송 경로 제한

현재 Native의 Early IPC Descriptor는 P가 모든 이미지를 소비한다는 조건을 사용한다.
이 경로를 Token Split에 그대로 적용하면 미사용 Descriptor가 남을 수 있다.
Token Split에는 이미지별 실제 소비자 범위와 독립적인 IPC Descriptor가 필요하다.
그 구현과 Lifetime 검증 전에는 Native 전용 Early IPC 경로를 Token Split에 적용하지 않는다.
