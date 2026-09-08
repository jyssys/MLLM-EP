# Prior-art refresh during deferred GPU completion

## ACE — new adjacent collision, not a fourth reproduced baseline

Primary source: [ACE, arXiv2609.05228v1](https://arxiv.org/abs/2609.05228),
submitted2026-09-04. Downloaded PDF SHA256
`baac4d38140340518c8383c5def205ea3dd3eed53bab778a84dc666fd1313dde`.
Read main methods/evaluation/conclusion pp1–8; appendix not fully audited.

- GSP estimates coupled SwiGLU/RMSNorm response capacity from checkpoint weights.
- RCR uses centered router-weight directions and offline expert responses.
- Online scoring combines router gates with both tables, protects top1 and
  renormalizes surviving gates. This is not MoDES's unrenormalized threshold rule.
- Important scope distinction: estimators are data-free, but evaluation uses an
  unlabeled full-model score pass per workload to map realized skipping budgets.
  The conclusion explicitly leaves threshold transfer, repeated-run uncertainty
  and distributed dispatch for future study.
- Evaluation is text-only LLMs and A100 inference, not a demonstrated Qwen-VL
  EP4 successor. We do not import its speedups into this project's measurements.
- It directly weakens a generic proposed contribution of replacing calibration
  with parameter-derived contribution scores. It does not by itself settle a
  new causal MLLM/EP limitation.

Adversarial caution: the text says all comparison methods retain original top-k
candidate sets and uses an executed-slot skipping budget. Original SERE can
reroute a slot to another token's batch-primary expert outside that token's top-k,
and reduces active-expert working set rather than necessarily slot count. Without
auditing ACE's baseline code, its SERE numbers are not evidence that faithful
SERE fails. Our unchanged official rerouter remains the baseline.

## Previously audited collisions rechecked

[PROBE](https://arxiv.org/abs/2602.00509) combines a gate-initialized learned
lookahead predictor, bounded replication planning and split-phase transmission.
"Improve Libra's predictor" alone remains insufficient novelty.

[AnyExperts](https://arxiv.org/abs/2511.18314) already distinguishes semantic
importance and OCR/NLP sensitivity. Its trained real/virtual allocation is not
the same algorithm as training-free MoDES, but a raw OCR sensitivity observation
is not new. [MoDES's official repository](https://github.com/ModelTC/MoDES)
still lists the fast CUDA layer implementation as a TODO; our DeepEP-sentinel
path is an explicitly labeled faithful-math deployment port, not official speed
reproduction.

Search also surfaced a third-party summary of an August REDtech/SERE deployment.
No primary REDtech text was obtained in this refresh; therefore none of its
reported performance numbers are used as evidence or a decisive novelty veto.

## Original papers themselves are adversarial controls

Libra Appendix D explicitly targets prefill-heavy TTFT and acknowledges that
short sequences can be slower when the local-MoE window cannot cover CPU work.
Its principal models are235B/355B on8×H200, not30B on4×H100. Therefore the small
native-Qwen slowdown is not a newly discovered MLLM failure. The bounded large
text control and current-route-frozen prediction oracle must distinguish scale
from an actual limitation in the supplied prediction/replication mechanism.

MoDES Figure6 measures a single H200, with prefill batch8 and decode context1024.
The discussion already attributes lower decode gains partly to smaller text-token
skipping ratios. It also reports the Qwen ChartQA trade-off85.08→78.84 at its
aggressive operating point. Neither a prefill/decode asymmetry nor OCR loss alone
passes the successor novelty gate. Our GQA-calibrated targets and EP4 request
timing are additional deployment evidence, not a reproduction of Figure6.

SERE AppendixC.2 says that rerouting does not reduce FLOPs and does not promise
prefill acceleration. Its batch-primary union is explicit. A fresh observation
that larger batches dilute aggressive substitution is scientifically useful as
a control, but is not by itself a hidden assumption omitted by the paper.

Source: locally downloaded full texts `references/{libra,modes,sere}.txt` under
the result root; primary links are in each paper audit. These checks were made
before final candidate ranking, not after selecting a desired positive result.
