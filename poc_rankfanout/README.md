# Rank-Fanout-Aware EP Communication PoC

This directory is an isolated PoC for testing whether destination-rank
fanout changes the relative cost of vLLM's AllGather+ReduceScatter (AGRS) and
DeepEP high-throughput (HT) expert-parallel communication paths.

The working contract is [SPEC.md](SPEC.md).  The PoC does **not** implement
dynamic backend switching.  It provides:

- exact EP4/top-8 route controls whose token count, expert histogram, and
  rank-load vector are identical across fanout 1/2/3/4;
- full-path and bounded operator benchmarks for AGRS and DeepEP HT;
- Qwen3-VL route/fanout capture;
- strict aligned best-static and perfect-oracle analysis;
- an Amdahl-adjusted TTFT upper bound and GO/HOLD/NO-GO decision.

All GPU launchers refuse to run unless
`CUDA_VISIBLE_DEVICES=4,5,6,7`.  CPU/GPU expert offload, EPLB, replication,
weight movement, token merging, and dynamic switching are out of scope.

## CPU checks

```bash
python -m pytest poc_rankfanout/tests -q
python poc_rankfanout/scripts/generate_synthetic_routes.py \
  --output /tmp/rankfanout_routes.npz
```

GPU commands and the exact environment pin are written to the result root by
the launchers.  See `reports/final_report.md` for the evidence boundary.

## Main entry points

- `scripts/check_runtime_path.py`: installed-source contract and hashes.
- `scripts/bench_synthetic_backends.py`: controlled F1–F4 operator matrix.
- `scripts/bench_real_static_backends.py`: Qwen3-VL static serving and direct
  router hook capture.
- `scripts/bench_real_route_replay.py`: bit-identical real-route replay.
- `scripts/analyze_synthetic.py`, `analyze_real_fanout.py`,
  `analyze_clean_static.py`, and `analyze_route_replay_oracle.py`: compact
  evidence tables and figures.

The completed raw result root is
`results/rank_fanout_ep_communication_poc_20260910_192050/`. It remains local
and ignored by default; compact derived tables/figures are committed. The
final decision and exact evidence boundaries are in
`reports/final_report.md` and `reports/decision.json`.
