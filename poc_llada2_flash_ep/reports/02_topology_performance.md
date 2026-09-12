# Topology performance

## Compared contracts

The official dInfer command is TP4. The sparse comparison is a bounded,
quality-validated **dense-TP4 + routed-EP4** contract: 64 complete routed
experts per rank, DeepEP normal-mode dispatch/combine, BF16 Triton local
experts, replicated shared experts on disjoint source rows, and an exact
source-row gather. Both use the same four physical H100s, checkpoint, decoder,
prompts, dtype, warmup, and timing boundary.

An orthogonal TP2×EP2 mesh is not supported by the executable SGLang 0.5.3
manual-runner path. Current SGLang main contains materially newer dLLM/DeepEP
support but requires a CUDA-13-era stack unavailable on this driver. A fake
hybrid collective was not constructed.

## Capacity and bounded quality

| Metric | TP4 | routed EP4 |
|---|---:|---:|
| Peak observed HBM/rank | 56.999 GiB | 56.862 GiB |
| Routed expert ownership | 256 tensor-sharded | 64 complete experts |
| Remote assignment fraction | n/a | 74.97% |
| GSM8K-32 score, each of five restarts | 5/32 | 5/32 |
| HumanEval-32 score, one screen | 4/32 | 4/32 |

The 205.8 GB BF16 checkpoint cannot fit on one 80 GB H100. Unlike the earlier
SDAR study, EP is a capacity-relevant decomposition here even though it is not
automatically the latency winner.

## Longer quality-valid run

GSM8K-32, generation budget 128, five paired independent engine restarts:

| Restart | EP wall gain vs TP | forward-normalized gain | throughput gain |
|---:|---:|---:|---:|
| 1 | +29.51% | +27.42% | +41.87% |
| 2 | -27.42% | -31.21% | -21.52% |
| 3 | -33.80% | -37.78% | -25.26% |
| 4 | +8.02% | +5.28% | +8.71% |
| 5 | -54.45% | -59.05% | -35.26% |
| paired median | **-27.42%** | **-31.21%** | **-21.52%** |

Unpaired medians are 34.396 s TP4 and 38.396 s EP4. The sign reversal across
restarts is the important result: it rules out a stable EP4 latency advantage
on this shared node. GPUs 4--7 were continuously occupied by unrelated jobs;
although they were never touched, their shared host/NVSwitch traffic is a
plausible common-state confound. This does not invalidate ownership or
operator measurements, but it prevents a topology winner claim from a single
best run.

## Conclusion

True EP4 works and is quality-valid, but the evidence does not identify a
stable latency winner over TP4. The result is a topology characterization, not
a dynamic-topology method opportunity.
