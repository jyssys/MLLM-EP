# Resume state

The user reduced the current pilot to GSM8K 128 and HumanEval 32 questions.
Do not start a duplicate controller.
The pilot controller 817585 completed the reduced baseline.
The trace collection controller 847014 completed all 16 requests.
The old controllers 704342 and 736119 stopped after the user changed the scope.
Check process ownership and command lines before any signal.

## Current execution

Group A uses GPUs 0–3 for physical TRUE EP4.
Group B uses GPUs 4–7 for four independent official HF workers.
Both paths use the exact non-preview Mini model revision.
Their decoder and KV protocols differ.
Do not interpret their timing ratio as EP-specific acceleration.

The reduced baseline reused completed generations for GSM8K IDs 0–127.
It evaluated HumanEval IDs 0–31 in both backends.
The completed trace queue covered the first 16 frozen non-test validation trajectories.
Group A receives even manifest offsets. Group B receives odd manifest offsets.
All runs retain every expert edge and assert no-op and observer output parity.
The confidence observer records only values from earlier completed decode forwards.
Its additional softmax and tracing overhead exclude these runs from primary latency claims.
All 16 requests passed those parity checks.
The causal confidence analysis is complete.
Its AUROC concerns commitment, not marginal expert importance.
The next experiment must test controlled expert-contribution omission on non-test validation requests.
Dense semantic emulation must not count as actual expert work reduction.
Do not claim a speedup until physical sparse execution and clean timing pass their gates.
Completed requests remain in the raw artifact directories.
The raw records must not be rewritten.

## Deferred original baseline follow-up

The original follow-up controller no longer executes.
The following paragraph describes the deferred original plan, not an active queue.
That controller waited for the full quality controller to exit.
It scores HumanEval inside the existing namespace sandbox.
It requires all four full benchmark cohorts before further baseline checks.
It then executes exact P0/no-op comparisons in isolated EP4 groups.
The server-start order is A0, B0, B1, A1.
Each start uses ten frozen validation prompts and three timing repetitions.
It also collects separate two-request validation traces in each group.
It restores owned burners after these baseline checks.

This queue does not implement or approve approximate policies.
Its terminal state is `BASELINE_COMPLETE_AWAIT_PHASE2_REVIEW`.
That state does not mean the full PoC is complete.
The active pilot ends at `PILOT_BASELINE_READY_REVIEW` instead.

## Required scientific review

1. Review pilot accuracy, Wilson intervals, termination, and paired differences.
2. Review Group B physical ownership and no-op parity.
3. Review isolated P0/no-op overhead with request-clustered uncertainty.
4. Inspect real validation traces and observer overhead.
5. Test causal refinement sensitivity on the frozen training partitions.
6. Fit physical EP costs and compare equal-edge counterfactuals.
7. Apply the 5% E2E and 2% incremental EP headroom gates.
8. Implement physical sparse policies only if those gates pass.

## Logging corrections

The initial native return removes EOS tokens.
The initial logger therefore misclassified short returns as length-capped.
The initial native NFE counter also undercounts chunked prefill calls.
`postprocess.py` derives both corrections from audited source behavior.
It preserves the original fields and labels source-derived values.
New worker starts capture terminal tokens and count model forwards directly.

The initial worker driver source snapshot was not captured at launch.
Do not invent a launch-time source hash for that driver.
Pinned model, helper, and active dInfer source hashes remain available.
New worker starts also save exact driver source snapshots.

## Commands

```bash
cd /home/esjung/MLLM-EP-github
python -m poc_flashvep.refinement_ep_allocation.resources status
python -m poc_flashvep.refinement_ep_allocation.postprocess --pilot --score-humaneval
python -m unittest poc_flashvep.refinement_ep_allocation.test_contract -v
```

The branch is `flashvep/refinement-aware-ep-expert-allocation-poc`.
The first published checkpoint is `01adbfae265c8e827a49f36c232c8df802e7f3dd`.
Preserve all unrelated staged and dirty work.
Do not resume the stopped Looped MoE study.
