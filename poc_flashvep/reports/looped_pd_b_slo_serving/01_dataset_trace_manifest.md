# Real-text workloads and arrival traces

The source is [ShareGPT Vicuna unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
The dataset card declares Apache-2.0. The provenance JSON records its pinned revision and downloaded-file SHA-256.
This filtered English dataset is not an unbiased production population.
No repetitive synthetic filler is used.

The preparation shuffles conversations with seed202610291.
It selects one user turn per conversation and retains all preceding dialogue.
Each model applies its original chat template with a generation prompt.
The filter requires 32–3072 prompt tokens for both model tokenizers.
The entire input plus the largest output ceiling stays within Huginn's 4096-position configuration.
No prompt is silently truncated to meet that filter.

| Set | Distinct prompts | Purpose |
|---|---:|---|
| Calibration | 24 | Baseline-only SLO calibration |
| Correctness | 40 | Sequential PD_G1/B token and depth parity |
| W1 | 200 | Primary mixed-length natural conversation serving |
| W2 | 100 | Actual long conversation/code/document text, at least 1024 tokens in both tokenizers |
| W3 | 100 | 80 short and 20 long requests with predeclared transient bursts |
| W4 | 80 | Moderate prompts with output ceiling256 as a decode-heavy negative control |

Calibration, correctness, and W1 prompt hashes are disjoint.
W2/W3/W4 use a separate remainder of the same source pool.
The supplementary sets can share some source conversations with each other. They do not replace the primary W1 population.
W2 is a long-dialogue subset, not a separately curated document-summarization benchmark.
W4's long output ceiling does not force long output. Actual EOS and output distributions remain required.

`data/*.json` preserves prompt IDs, messages, model token IDs, token lengths, and prompt hashes.
`configs/dataset_provenance.json` preserves the selection and source metadata.
Each `raw/<run>/arrival_trace.json` preserves exact prompt order, seed, arrival offsets, and window length.
Poisson traces are deterministic for a fixed seed, count, and rate.
W3 uses a 3x arrival state during seconds30–40 of every40s cycle.
The runtime does not use future model outputs or GPU durations to select an arrival or an allocation.

The workload is real-text, bounded-generation serving unless natural EOS completion is sufficiently frequent.
The output caps are W1=128, W2=64, W3=128, and W4=256.
An output cap is not an observed mean output length or evidence of answer completeness.
