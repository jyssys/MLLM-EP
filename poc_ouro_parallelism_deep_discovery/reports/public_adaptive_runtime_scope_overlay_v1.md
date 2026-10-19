# Public adaptive-runtime scope overlay v1

Date: 2026-10-19 UTC. This overlay adds source evidence. It changes no earlier report or measured result.

## Wording correction

[dynamic_runtime_prior_boundary.md:11](dynamic_runtime_prior_boundary.md:11) calls CDB “the public executable original-gate baseline.” Do not read that phrase as an exclusive public-code claim.

CDB remains the campaign's validated original-gate runtime. A second public implementation exists in vllm-rlt at revision `9d601ceaf9af63121c742df5f584467e23630a0b`. Its synchronous ordinary `ouro` path uses checkpoint gate tensors and depth-aware LAST_EXITED KV. Its request controls can express threshold 0.5 and minimum/maximum depths 2/4. This is source support, not campaign validation. [Pinned model](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/models/ouro.py#L223), [decision path](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/engine/llm_engine.py#L420)

The replacement wording is:

> CDB is the campaign's validated public original-gate runtime. Other public implementations exist, including vllm-rlt. Their configuration support, numerical behavior and measured scope need separate qualification.

## Limits that remain

The inspected RLT distributed path uses full-model P/D workers, not TP2/TP4. Its Base2.6 implementation is dimension-driven but explicitly unqualified. Ordinary adaptive `ouro` does not use its async path. Gate cast order differs from pinned CDB. These facts prevent automatic numerical or timing parity claims.

PR30 uses threshold 0.2 and changes stopping policy during feature stacking. PR31 compares replicas and P/D pools with async execution. Neither supplies a campaign-matched adaptive TP2/TP4 result. See the [full audit](vllm_rlt_runtime_source_audit_v1.md) and [source ledger](../artifacts/vllm_rlt_runtime_source_observations_v1.json).

This review installed and executed no runtime. The campaign baseline, candidate status, quality evidence and protected experiment remain unchanged.
