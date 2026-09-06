# Autonomous EP research discovery v3

This directory contains the bounded discovery driver/analysis used in the
fresh Qwen3-VL online traces. It does not change model math or routing and
does not implement an optimization method.

## Reproduction

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONPATH=/home/esjung/MLLM-EP-github/poc_flashvep/online_routing_geometry/hooks:/home/esjung/MLLM-EP-github:$PYTHONPATH
python poc_flashvep/online_routing_geometry/run_online.py --model <local-qwen3-vl-snapshot> --out <result-dir> --concurrency 8 --waves 32 --warmups 3 --max-tokens 2 --max-batched-tokens 8192 --backend deepep_high_throughput
python analyze_discovery.py --root <result-dir> --out <result-dir>/atlas_measured --measured-only
```

For the shape-state diagnostic, the bounded driver also supports
`--warmup-slot`, `--fixed-slot`, and `--alternate-slots 0,3`.  The
`analyze_state_transition.py` utility summarizes those controls separately;
they are not silently folded into a method speedup claim.

The parser collapses TP/EP worker rows by DP/local invocation/layer and takes
the maximum same-DP span. It explicitly excludes warmup rows when
`--measured-only` is selected and never subtracts timestamps from different
GPUs.
