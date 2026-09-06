# Runtime source audit

Validated local versions:

- vLLM: `/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm`, 0.20.0+cu129, V1.
- DeepEP: `/home/esjung/.venvs/flashvep-deepep-v020/lib/python3.12/site-packages/deep_ep`, 1.2.1+73b6ea4.

The Qwen3-VL MoE path selects `DeepEPHTPrepareAndFinalize` from
`vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`.  It
calls `dbo_get_previous_event(self.buffer.capture)` before
`get_dispatch_layout`/`dispatch`, waits for the dispatch event before local
expert execution, and passes a second previous event to combine.  The
installed C++ `EventHandle` has no nonblocking query exposed to Python; with
DBO disabled the helper returns `None` in the observed path.  The actual
DeepEP kernels nevertheless carry the notify/barrier dependency shown by the
prior Nsight child-worker trace.

The local hook wraps only measurement boundaries and records same-device
CUDA events.  The added context-file handoff makes the active wave visible in
engine child workers without changing model math or scheduling.  It does not
claim exact per-request token attribution for co-batched invocations; the
analysis caps shared excess at each request's observed E2E.
