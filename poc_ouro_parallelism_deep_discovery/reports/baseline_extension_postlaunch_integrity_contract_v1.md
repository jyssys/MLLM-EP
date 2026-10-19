# Baseline extension: post-launch integrity supplement

## Reason and timing

An independent source review found admission gaps in the first analyzer.
Synthetic changed prompts, unbound text, and unsupported stop metadata passed its request-level helper.
The analyzer also omitted the frozen launch anchor.
These were synthetic counterexamples. They did not establish a discrepancy in a generated record.

This supplement was added after launch and after the first complete arm.
It is not a prelaunch analysis freeze.
It changes no answer extraction, grading, sample membership, token cap, sampling, gate, EOS, or stop rule.
The frozen grading sources remain hash checked.
The original runner and launch sources remain unchanged.

## Admission checks

1. Require successful runner and outer guard status.
2. Check all launch source hashes and the frozen64-request protocol.
3. Load each pinned local tokenizer with the same Transformers5.9.0 version.
4. Re-encode each frozen prompt with `add_special_tokens=False`.
5. Require exact equality with recorded prompt IDs.
6. Decode recorded output IDs with `skip_special_tokens=False`.
7. Require exact equality with recorded text.
8. Apply the unchanged finish precedence: EOS, then8192-token cap, then frozen stop.
9. Require an actual nonempty encoded stop suffix for stop completion.
10. Reject output after an earlier EOS or frozen stop.
11. Hash all64 raw records in each complete arm.
12. Require those exact hashes before the scoring analyzer admits the arm.

The tokenizer check does not re-encode decoded output text.
Different tokenizations can decode to the same text.
The raw output IDs remain authoritative.

## Evidence boundary

The validator evaluates no answer correctness.
It checks provenance and recorded stopping behavior, not independent model re-execution.
The analyzer still requires source-counter and depth-event agreement.
Incomplete arms remain diagnostic evidence only.
No performance or quality row can silently exclude an unexecuted request.

The new synthetic CPU tests exercise the earlier gaps and valid EOS/stop cases.
Existing analyzer tests remain unchanged. They test the narrower helper, not complete final admission.
Earlier analyzer checkpoints remain intact.
No old score or raw file is rewritten.

## Reproduction

Run the validator in the isolated Python3.14 environment with no visible CUDA device.
The script imports the existing tokenizer packages. It loads no model weights and starts no GPU job.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.validate_baseline_extension_integrity \
  --output poc_ouro_parallelism_deep_discovery/artifacts/baseline_extension_integrity_COMPLETE.json

/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.analyze_baseline_extension \
  --integrity poc_ouro_parallelism_deep_discovery/artifacts/baseline_extension_integrity_COMPLETE.json \
  --output poc_ouro_parallelism_deep_discovery/analysis/baseline_extension_COMPLETE
```

Use a new output path for each checkpoint.
The example names do not claim that all arms were complete when this contract was written.
