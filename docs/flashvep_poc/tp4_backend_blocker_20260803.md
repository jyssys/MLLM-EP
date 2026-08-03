# Debate Note: TP4 MoE Backend And Collective Blockers

Date: 2026-08-03 (Asia/Seoul)

## Outcome

Qwen3-VL-30B-A3B-Instruct loads on physical GPUs 4-7 with TP=4 and effective
EP=4. The vLLM automatic MoE backend selects FlashInfer CUTLASS Unquantized,
but the engine does not complete its initial language-model dummy forward.
No request output or result JSON is produced; the stalled runs were manually
interrupted without killing unrelated GPU processes.

The smallest successful change is explicit `moe_backend=triton`. With that
setting the 224x224 smoke produces token 1986 (`This`), 79 prompt tokens, 64
visual tokens, and routed-expert tensor shape `[79,48,8]`. Each physical GPU
4-7 uses approximately 18.95 GiB. Evidence is in
`poc_flashvep/results/baseline/smoke_tp4_gpu4567_triton.json`.

## Reproducer

The common command boundary was:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=. \
/home/esjung/anaconda3/envs/flashvep-poc/bin/python \
  scripts/vllm_ep_sanity.py \
  --model-path /home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c \
  --tensor-parallel-size 4 \
  --kv-cache-memory-bytes 1073741824 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --skip-mm-profiling \
  --output OUTPUT.json
```

With the default `--moe-backend auto`, logs reach FlashInfer CUTLASS backend
selection and the startup dummy forward but do not complete. Disabling the
custom all-reduce did not restore progress. Adding `--moe-backend triton`
completed. This is a backend substitution, so the successful run is not
silently labeled as the original auto-backend performance baseline.

The first online model fetch downloaded all 13 shards (57.87 GiB) but retained
an open Hugging Face/Xet operation. Subsequent work uses the exact resolved
snapshot path with both offline environment flags. Model weights were not
modified.

## More Important Structural Finding

For this TP4/EP4, DP1 configuration, vLLM reports
`MoEPrepareAndFinalizeNoDPEPModular`. Effective EP is flattened over the TP
ranks: every rank receives the full token sequence and owns 32 local experts.
There is no dispatch All-to-All. The final MoE result uses a tensor-parallel
all-reduce.

Therefore the configured string `allgather_reducescatter` is not evidence that
this request traverses those two collectives. Phase 1 records the actual path
as:

```text
router -> local prepare (no collective) -> local experts
       -> local finalize -> TP all-reduce combine
```

This layout is runnable but is not a faithful exercise of FlashVEP's intended
dispatch/expert/combine All-to-All pipeline. Any future debate should treat
that as a design blocker, independently of the backend hang.

## Debate Options

1. Diagnose the FlashInfer CUTLASS TP4 startup hang, then repeat Phase 1 with
   the stronger backend before reconsidering headroom.
2. Explicitly authorize a DP-based EP arrangement that invokes actual EP
   dispatch/combine collectives, accepting changed offline request semantics.
3. Keep the current TP-derived EP layout and stop FlashVEP work because the
   communication target is absent.
