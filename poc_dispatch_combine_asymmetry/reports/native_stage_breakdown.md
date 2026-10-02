# Fresh native TRUE EP4 stage baseline

No past GPU2/3/6/7 absolute timing was reused. Warmup3 per arm/size; final10 repeats per arm/size. VL visual/text alternate order within repeat; LLM is a separate model-loading block, so VL-vs-LLM is NOT a same-engine paired experiment.

| Arm | tokens | Dispatch ms/layer | Expert | Combine | mean request D/C ratio |
|---|---:|---:|---:|---:|---:|
| VL visual low | 175 | 0.9131 | 0.4020 | 0.1131 | 8.085 |
| VL visual medium | 721 | 0.9305 | 0.4507 | 0.0826 | 11.339 |
| VL visual high | 2029 | 1.1092 | 0.5264 | 0.1246 | 9.051 |
| VL text low | 174 | 0.4703 | 0.3973 | 0.1249 | 3.796 |
| VL text medium | 724 | 0.5498 | 0.4298 | 0.1219 | 4.747 |
| VL text high | 2033 | 0.5184 | 0.5114 | 0.1331 | 3.927 |
| Qwen3 LLM text low | 174 | 1.4125 | 0.4005 | 0.4538 | 3.114 |
| Qwen3 LLM text medium | 724 | 1.4733 | 0.4376 | 0.4471 | 3.296 |
| Qwen3 LLM text high | 2033 | 1.4405 | 0.5070 | 0.4140 | 3.559 |


Each repeat selects ONE rank maximizing its cumulative 48-layer Dispatch+Expert+Combine. All reported components come from that rank; no independent component maxima or unrelated rank averages. These are sum/mean of local prepare/expert/finalize intervals, NOT a globally barrier-composed end-to-end latency. 48 layers are not treated as independent samples. Mean, median, P90 and repeat-bootstrap95% CI (2000) in final_metrics.json.

High same-model visual−text Dispatch = 0.5909 ms/layer, CI [0.5653875966385629, 0.614917141327945]; paired percentage 115.33%, CI [105.66803633839672, 124.5769221574291].
Combine delta -0.0085 ms/layer, CI [-0.020241297431445378, 0.0001179974328260793].
Visual−LLM Dispatch = -23.00% (unpaired model blocks); it does NOT exceed the standalone LLM.

Token scaling is descriptive, one image/text prompt per bin, not a fitted latency predictor. Actual text counts174/724/2033 versus visual175/721/2029 are explicitly retained.
