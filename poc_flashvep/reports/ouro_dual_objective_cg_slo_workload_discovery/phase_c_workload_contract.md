# Prospective natural workload contract

The model is Base Ouro-1.4B at the pinned revision, BF16, greedy Static R4.
Every real prompt and output token executes all four recurrent passes.
The input lock contains 473 variants from 388 source families.
No source family crosses TRAIN, VALIDATION, CALIBRATION, or FINAL.
Long/stress variants of one article stay together.

CHAT uses public Dolly instruction/context requests near the 256-token input target.
LONG uses natural arXiv excerpts near the 1,800-token target.
LONG8K_STRESS uses genuine long article excerpts near 8K, without repeated filler.
The finite12-request MIXED cohort uses nine chat and three long requests, or75%/25%.
This is a rounding departure from the 70%/30% target, not an exact SLOWeave reproduction.
BURSTY uses the same finite mix with a registered two-state arrival process.
Targets describe input construction and output intent, not forced generated lengths.
Actual prompt/output distributions, EOS, and caps remain mandatory results.

Serving uses native EOS and the official tokenizer's chat-stop token.
The initial validation cap is 1,024. The frozen FINAL cap is 4,096 for every arm.
The prospective amendment triggered because the largest validation cohort cap fraction was 25%, above the 5% trigger.
The amendment occurred before FINAL outcomes. Old capped outputs remain unchanged.
Capped responses cannot pass complete-response SLO attainment.
Quality uses the preserved few-shot GSM8K/MATH prompts and a separate 4,096-token cap.
Serving prompts do not substitute for those accuracy questions.

The initial validation lane has concurrent chat-like decodes and one delayed long prefill.
It is a finite natural cohort, not a sustainable-rate result.
All objective controls cross on the same GPU/model-process block.
Independent starts rotate physical GPUs4–7 and policy order.
No GPU0–3 process is started or stopped by this campaign.

SLO sensitivity must distinguish every-gap request attainment from pooled token P99.
Final rate seeds, offered-load brackets, objective weights, and SLO tiers must freeze before FINAL results.
Qualified capacity requires at least 1,000 offered requests per arm/rate and three independent starts.
An unstable queue, a cap, or an unfinished response cannot qualify as successful service.

The workload families are inspired by [SLOWeave](https://arxiv.org/html/2609.07883v1), not its model/runtime reproduction.
The paper uses synthetic length distributions and a within-request P99 TPOT goodput rule.
This study additionally reports the stricter every-gap complete-response criterion.
[P-PAS](https://arxiv.org/html/2608.15171v2) motivates pressure-dependent budget controls, not a claimed official RLT implementation.

## Frozen FINAL execution

The primary objective is O6, weight 2, fixed single-quantum budget 120ms.
The controller's exposure normalization uses 200ms throughout.
Postprocessed SLO thresholds do not retune that controller.
The program registers30 four-worker characterization jobs and six serial confirmation jobs.
Each policy block uses 12 offered natural requests. Each workload/rate cell has three independent starts.
The two ordinary arrival parameters are 0.25 and 0.9 requests/s.
Bursty uses rates 0.5× and 2× that parameter with 8-second deterministic dwell periods.
Its high/low ratio is 4. The realized offered rate is recorded separately.
The 8K stress lane uses seven initial short requests, a delayed long request, and four later short requests.
It is a hybrid arrival diagnostic, not a pure Poisson workload.

Quiet source families are disjoint from characterization source families.
Inputs can repeat across registered rate/workload contrasts within one partition.
Those repeats are not independent source samples.
The quiet lane reserves GPUs4–7 for one measurement worker at a time.
Pre-existing GPU0–3 background processes remain untouched.
The source/arrival lock must not change after these runs start.

Serving clocks start at registered in-process arrivals of pretokenized inputs.
This scope excludes HTTP transport and prompt tokenization.
Warm-up and model construction occur before the serving window.
Lazy graph work that occurs inside that window remains in measured latency.
