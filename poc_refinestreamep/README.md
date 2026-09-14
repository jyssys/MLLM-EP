# RefineStreamEP serving-first PoC

This artifact tests legacy DeepEP V1 Normal and low-latency paths under
concurrent, non-stationary LLaDA2 refinement route streams on physical GPUs
4--7. The decision is `NO-SERVING-GAP`; no custom kernel was implemented.

Key files:

- `M_CROSSOVER.csv`: same-rank-critical Normal/LL transaction medians.
- `CAPACITY_MATRIX.csv`: actual-M versus per-source-rank LL capacity.
- `INFLIGHT_RESULTS.csv`: three-restart Q=1--16 results.
- `REFINEMENT_STREAMS.jsonl`: chronological measured-route stream.
- `SERVING_RESULTS.csv`: five-seed route-replay serving rows.
- `SERVING_ORACLES.csv`: all serving policies and structural oracles.
- `reports/final_decision.md`: decision and evidence boundary.

GPU scripts hard-fail unless `CUDA_VISIBLE_DEVICES=4,5,6,7`. Aggregate and
serving analysis can be regenerated with:

```bash
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python \
  poc_refinestreamep/scripts/analyze_and_simulate_serving.py \
  --root poc_refinestreamep
```
