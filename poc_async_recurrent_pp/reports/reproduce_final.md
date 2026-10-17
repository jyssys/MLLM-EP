# Final evidence reproduction

Run these commands from `/home/esjung/MLLM-EP-async-recurrent-pp`.
The commands use only this PoC's files.
Use a new output directory for a repeated analysis.
The scripts refuse an existing destination where they create final evidence.

## CPU tests

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest discover -s poc_async_recurrent_pp/tests -v
```

## Independent final scoring

The source wave must finish successfully for all four arms.
The scorer reads raw outputs without changing them.
It applies the same parser version to every arm.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.derive_final --wave poc_async_recurrent_pp/outputs/pilot_cap8192_v2
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.evidence_audit --wave poc_async_recurrent_pp/outputs/pilot_cap8192_v2 --output poc_async_recurrent_pp/outputs/validation/evidence_final.json
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.final_figures --analysis poc_async_recurrent_pp/outputs/final_analysis
```

The final scorer requires complete paired question IDs and identical prompt tokens.
It checks the first generated token because every arm uses exact prefill.
It preserves original grades and records any parser-related grade change.
The primary endpoint is answer accuracy.
Completed-answer accuracy is a separate endpoint that also requires EOS.

The bootstrap uses 10,000 paired question resamples with seed 20261017.
Full canonical benchmark evaluation remains necessary for any quality GO claim.
A pilot can reject a large quality regression without establishing full-benchmark accuracy.

## Calibration-only margin probe

This command stops only a verified burner on physical GPU 4.
The guard restores the burner after completion or failure.
Do not run it against an active evaluation or an unidentified GPU context.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.gpu_session --gpus 4 --output outputs/margins_v1 --timeout 1800 -- /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.margin_probe
```

The probe is a teacher-forced calibration diagnostic.
It does not train or select a predictor.
It does not measure distributed latency.

## Publication check

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.audit_publication --output poc_async_recurrent_pp/outputs/validation/publication_scan_final.json
```

This check compares candidate files against credential values in the current environment.
It also checks archive members and a conservative 100,000,000-byte file ceiling.
It does not prove that every possible secret is absent.
Private GPU-process inventories remain local and ignored by Git.

## Device and model names

The four quality workers are independent single-GPU numerical emulators.
Physical GPUs 4, 5, 6, and 7 evaluate Exact, Stale, Delta, and W0 respectively.
Each worker sees its assigned physical GPU as logical `cuda:0`.
This execution is not PP4 or a distributed architecture throughput benchmark.

The Python environment name contains `llada2-mini-reference`.
The loaded checkpoint is nevertheless the pinned `ByteDance/Ouro-1.4B-Thinking` model.
The environment name does not select the model.
A restored `VLLM::EngineCore` is the requested utilization burner, not an Ouro pipeline stage.
