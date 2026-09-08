# SERE candidate — causal evidence and limits

This records the strongest diagnosed quality limitation,not a promoted successor.
All-three screening is required before final ranking; final rank is in the main
report and scoreboard.

## Established controls

1. The official400×128 BF16-norm calibration and unchanged CUDA rerouter replace
   the exploratory FP32-Gram table. Large B1 quality losses remain in HF and the
   separate actual EP port. An arithmetic-only explanation is insufficient.
2. Holding the held-out questions,images,scorer and vanilla comparison fixed,
   S2/rho.5→S4/rho.5 removes most observed loss. This is a real intervention,
   but an existing paper parameter rather than a new method.
3. Fixed HF B16 cohorts rescue17/17 ChartQA and11/12 GQA newly wrong B1 cases
   within the subset that vanilla answers correctly at both sizes. Finished/
   padded HF rows remain in this cohort,so this is not interchangeable with
   natural online batch membership.
4. Actual online B1/B4/B16 repeats and scheduler traces distinguish offered
   concurrency from live DP-local membership. The source's batch-primary union
   can dilute substitution as more experts become primary. The paper already
   exposes this batch contract; it is not a newly discovered hidden mechanism.
5. Fixed32-token controls retain a slowdown; new long answers alone do not explain
   the port's cost. Algorithmically no-op S8 still incurs15.67% median request
   regression in a bounded B16 control. Through-first-EOS output agreement is
   96/96; forced post-EOS continuation is not uniformly bitwise identical.

## What is NOT established

- Modality is not isolated as the cause. Both visual conditioning and generated
  answer semantics can matter; no matched-cause ON/OFF modality experiment
  establishes an MLLM-specific law.
- Batch-size changes also change kernel arithmetic and membership. They are not
  an exact same-target/per-expert-histogram causal replay.
- The speed floor belongs to this vLLM/EP port,including source-local union across
  TP-sharded tokens. It does not refute the paper's single-H20 speed results.
- Kimi and a nontrivial successor are untested because the existing-knob attack
  already removes the main measured loss. Unknown generality is not failure.

Conclusion: aggressive substitution causes material finite-sample B1 errors,but
the tested successor premise fails the **nontriviality** and demonstrated-headroom
requirements. It is not evidence that SERE universally fails in MLLMs.
