# Environment and baseline readiness

Resolved authoritative spec: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/dispatch_combine_asymmetry_discovery_spec_gpu0145.md`. The requested basename without `_gpu0145` does not exist; this unique, matching specification was read in full and snapshotted. No earlier spec was substituted.

Base git SHA: `2553b64a7993ff681cafe63b7627b001f4bcd3d1`. Separate worktree and new branch; original staged/dirty changes preserved.
CUDA_VISIBLE_DEVICES=0,1,4,5; logical 0/1/2/3 map to physical 0/1/4/5.
H100 80GB HBM3 (81559 MiB each); allowed GPU pairs/all four are NV18 connected via NVSwitch.
TP groups=(0,1)/(2,3), DP peer groups=(0,2)/(1,3), EP group=(0,1,2,3), PP1. EP=TP×DP, four GPUs total.

MLLM revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`; LLM revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. Both actual configs: L48/d2048/m768/E128/K8, BF16, norm_topk_prob=true. All shards exist. HF storage layouts differ but actual loaded banks [32,1536,2048]/[32,2048,768] match; 9,437,184 BF16 bytes/expert. Actual expert maps verified on all 48 layers/all 4 ranks in each of 9 captured arm-size strata.
This matches MoE geometry, not weights or the entire attention/position architecture (VL MRoPE/DeepStack vs LLM RoPE).

vLLM 0.20.0+cu129, DeepEP 1.2.1+73b6ea4 (source SHA 73b6ea4a439ba03a695563f9fd242c8e4b02b37c), torch 2.11.0+cu129, CUDA 12.9; NCCL2.28.9. DeepEP HT + sparse Triton experts, eager, EP linear32/rank. EPLB/redundant experts/DBO/prefix cache off. Chunked prefill off, max_batched_tokens=20000/max_seqs=1, KV2GiB/rank, fixed one-token output. No environment upgrade or benchmark-quality reevaluation.

One real request is active on DP0; DP1 is idle/dummy, not a duplicate request. Source TP partitions the request; SP padding plus one dummy row on each idle source are separately accounted. Foreign loads on GPUs2/3/6/7 and shared host/NVSwitch are recorded but untouched. Only authorized user's 0/1/4/5 burner was stopped; all timing GPU PIDs belong to this PoC. Resource restoration uses existing run_utilize.sh with explicit VLLM_UTILIZE_GPUS=0,1,4,5 (script default is DIFFERENT).
Raw UUID/topology/owner/version evidence: artifacts/gpu_audit/environment.json.

After all measurements, the user requested GPU handoff and revoked burner restoration. The verified owned burner tree was stopped; final allowed-GPU compute PID snapshot is empty. No subsequent GPU run or burner restart. See resource_handoff.json.
