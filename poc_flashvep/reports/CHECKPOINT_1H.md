# T+1h checkpoint — online runtime-regime discovery

- Strongest positive: the communication-SM control is a real vLLM/DeepEP HT
  runtime knob (`VLLM_DBO_COMM_SMS`), with fresh 8/12/20 traces and stable
  sub-millisecond decode medians.
- Strongest negative: Model 2→3 fanout incremental RMSE remains approximately
  0% (slightly negative) on 123,456 fresh+prior online rows; fanout is not a
  new runtime signal.
- Open anomaly: per-shape/rank timing CV is high (median ~19.5%); the hook
  records full stock MoE intervals and cannot causally split dispatch/expert/
  combine in this serving path.
- Live data: three completed HT engine runs; LL feasibility attempts failed at
  DeepEP qp-depth assertion.  Next: finish burst trace, compare SMS envelopes,
  mine matched tails and complete oracle/decision artifacts.
