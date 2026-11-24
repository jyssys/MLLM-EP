# MLLM EPD Main100 Research Checkpoint

Status: ACTIVE. This is an incomplete research checkpoint, not a final serving result.
The user reduced future main experiments from Main400 to a frozen representative Main100.
The original Main400 and raw results remain unchanged. The subset does not preserve every tail statistic.
The model is Qwen/Qwen2.5-VL-7B-Instruct, original BF16, revision cc594898137f460bfe9f0759e9844b3ce807cfb5.
Main100 SHA256: 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1.

## Evidence limits

- Current serving controls use actual online encoding and installed vLLM EC/NIXL paths.
- ECExampleConnector uses CPU staging through per-run tmpfs. It is a debug connector, not production NIXL encoder transfer.
- Processor-result handoff is a common exact runtime repair. It is not Generic Token Split.
- Current Main100 timing cells are single-start diagnostics. No independent-start serving confidence interval is established.
- Full task quality and cross-geometry encoder gates remain unresolved. Complete source scorers match the latest reference point scores.
- Current physical GPU activity does not confirm P-prefill-heavy or safe additional prefill admission.
- Generic Token Split, inside-image split, online idle-aware split and E/D-safe extra prefill remain NOT_RUN or NOT_MEASURED.
- The CUDA IPC probe tests protocol correctness only. It is not an EC serving connector or an online speedup.
- Future work must preserve the strong baseline, original dataset, output and SLO contracts.

## Reproduction

The source module contains dataset,server,client,metric and figure generators.
Configs preserve commands,source hashes,model revision,arrival seeds and GPU identities.
The active manifest is configs/active_workload.json. The archived Main400 remains in data/.
archives/checkpoint_raw_records_v0.json lists lossless compressed top-level JSONL records.
Nested GPU profiles,tensor captures,images and full worker logs remain local at this checkpoint.
The final campaign must archive and verify remaining evidence,finish the gated scope and verify resource/Git closeout.
Source snapshots preserve exact Python bytes, including boot versions where available.
Installed Python wheel integrity does not establish native-library or upstream checkout equivalence.

## Resource state

Completed model and protocol jobs released their owned GPUs at safe checkpoints.
Approved burners run while GPUs are unused. This is an inter-phase restoration, not final campaign closeout.
