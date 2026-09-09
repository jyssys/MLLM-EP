# Official PP4 correctness sanity: EOS association bug

Status: root cause isolated; corrected fresh-engine short-answer regression
passed (see the chronological evidence below). Full task-quality equivalence is
not implied.

## Reproduction

The untouched official PP4 path generated `42` and `Paris` correctly, but
continued with extra dialogue despite `ignore_eos=False`. `check_short_answers.py`
fails both cases. Independent HF BF16 greedy output ends at EOS. Tokenizer input
IDs and EOS ID agree exactly between the isolated HF and SGLang environments.

The next teacher-forced diagnostic requests **raw special tokens**. Both normal
cached decode and prefill teacher forcing emit EOS 151645 at the correct point.
Therefore this is not a failure to compute the EOS logits or a PP hidden-state
corruption. The default text detokenizer hid EOS in the first smoke's output.

## Source attribution and intervention

`Req.check_finished` assigns `last_token_id = self.output_ids[0]`, while
`Scheduler.process_batch_result_decode` appends each new token to the end.
Consequently EOS is recognized only if it was the *first* output token.

`test_eos_regression.py` exercises the actual official Req class, appending
`[19, 17, 151645]` as the scheduler does. It fails before the change. The bounded
compatibility fix reads `output_ids[-1]`. The `ignore_eos=True` fixed-output
benchmark remains unchanged and has a separate regression control.

This is a trivial correctness fix, **not successor research evidence**. Keep the
official source pin plus patch; do not attribute shorter corrected requests to
FastPP scheduling gains. All strategy comparisons use the same corrected seam.

## Other diagnostics

- Installed PyNccl warmup hangs with this host's IB transport; disabling IB makes
  the official P2P-disabled launch work. Timing controls must use identical
  transport, and H100-appropriate P2P is a separate configuration control.
- Optional PP `return_logprob=True` fails because `TpModelWorker` returns a
  placeholder `LogitsProcessorOutput`. Avoid this endpoint capability in the
  benchmark. The temporary opt-in logits capture has been removed.
- The diagnostic restart used memory fraction 0.60 while NanoFlow workers were
  SIGSTOP-paused during CPU JIT. It is correctness-only, not timing evidence.
# Live regression after fresh engine restart

At 17:27 KST, `fastpp_eos_fixed_smoke_requests.jsonl` passes both sharp HF
short-answer tests: `42` and `Paris`, terminating after 3 and 2 tokens including
EOS respectively. Four requests complete. The optional sampling diagnostic is
absent from this engine. Full benchmark correctness still requires sequence
comparison across scheduling options; the short test alone does not certify it.

This engine shared GPU *residency* with SIGSTOP-paused NanoFlow workers and used
memory fraction 0.60. It is correctness evidence only, not benchmark evidence.
