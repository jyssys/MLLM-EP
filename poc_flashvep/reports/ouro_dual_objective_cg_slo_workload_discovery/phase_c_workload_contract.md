# Prospective natural workload contract

The model is Base Ouro-1.4B at the pinned revision, BF16, greedy Static R4.
Every real prompt and output token executes all four recurrent passes.
The input lock contains 473 variants from 388 source families.
No source family crosses TRAIN, VALIDATION, CALIBRATION, or FINAL.
Long/stress variants of one article stay together.

CHAT uses public Dolly instruction/context requests near the 256-token input target.
LONG uses natural arXiv excerpts near the 1,800-token target.
LONG8K_STRESS uses genuine long article excerpts near 8K, without repeated filler.
MIXED will use 70% chat-like and 30% long-source requests.
BURSTY will use the same length mix with a registered two-state arrival process.
Targets describe input construction and output intent, not forced generated lengths.
Actual prompt/output distributions, EOS, and caps remain mandatory results.

Serving uses native EOS and the official tokenizer's chat-stop token.
The primary natural-output cap is 1,024. Capped responses cannot pass complete-response SLO attainment.
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
