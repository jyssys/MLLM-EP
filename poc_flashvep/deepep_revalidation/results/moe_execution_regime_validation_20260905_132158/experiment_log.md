# Live experiment log

All commands were launched from `/home/esjung/MLLM-EP-github` with
`CUDA_VISIBLE_DEVICES=1,2,3,4`, vLLM 0.20.0 V1, TP2/DP2/EP4, BF16,
DeepEP high-throughput, TritonExperts, eager mode, five warmups, and shuffled
case manifests.  The replay driver was
`poc_flashvep/deepep_revalidation/vllm_backend_matrix.py`.

| Window | Cases / repetitions | Purpose | Result |
|---|---:|---|---|
| H1H2_seed17 | 36 / 30 | initial randomized M×fanout×active grid | complete, correctness true |
| H1H2_seed29 | 36 / 30 | independent order seed | complete, state outliers retained |
| H1_interleaved | 120 / 30 | M128/M512 sign-flip test | sign flip absent |
| H2_interleaved | 240 / 20 | active×fanout interaction | M512 penalty grows with active |
| H4_alignment_interleaved | 420 / 10 | tile/power-of-two boundary | no discontinuity |
| H4_focus_rep30 | 420 / 30 | M496–528 replication | smooth high-M increase |
| H5_layer4 / H5_layer44 | 60 each / 15 | layer persistence at M128/M512 | M512 persistent |
| H3_local_vs_deepep_retry2 | 40 / 10 | local expert versus DeepEP | both expert and dispatch contribute |
| H8_real_route | 11 cases / 10 | real route transfer | high-F natural routes, unmatched |
| H7_distribution_interleaved | 120 / 30 | per-expert distribution control | weak |
| H10_generic_qwen3 | 40 / 10 | generic text MoE check M128/M512 | same high-M direction |
| H6_geometry_interleaved / M1024 | 40 each / 20 | balanced geometry permutation | null |
| H1_boundary_interleaved | 260 / 10 | M64–1024 boundary curve | gradual onset, no sign flip |
| H1_M1024_rep30 | 60 / 30 | strong M1024 replication | expert +31.3%, wall +19.5% |
| H2_interleaved_rep30 | 360 / 30 | active×fanout replication | M512 expert +9.8–23.8% |
| H3_local_M1024 | 40 / 20 | local/DeepEP M1024 mechanism | local +40.5%, wall +12.2% |
| H8_real_route_layer44 | 11 cases / 10 | late-layer route transfer | one combine outlier; unmatched |
| H2_M1024_active | 120 / 20 | high-M active interaction | wall +15.4–24.3% |
| H5_layer44_M1024 | 20 planned / init failure | late-layer M1024 check | blocked; no value imputed |
| H10_generic_qwen3_M1024 | 20 / 10 | generic high-M replication | expert +32.4%, wall +2.1% |

Successful cases all carried route/token identity and correctness fields.  The
two failed H3 import attempts and the H5 layer-44 M1024 initialization failure
remain in their original directories and are not deleted or overwritten.
