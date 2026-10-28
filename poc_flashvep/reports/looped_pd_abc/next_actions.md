# Resume checklist

The active task is Dense Looped PD A/B/C, not an earlier MoE or diffusion task.
Do not resume the paused decode Loop PP project.
Do not terminate the active campaign when the user asks for status only.

1. Read the complete authoritative PD specification and the reporting skill.
2. Check `state/main_campaign.json`, `state/dispatcher.json`, and the recorded live PIDs.
3. Keep the current controller if it remains active.
4. Inspect new failed run statuses and their logs before any retry.
5. Use a new run name for a changed scientific configuration.
6. Preserve original failures and per-attempt source ZIPs.
7. Complete the three independent comparisons. Do not cancel B or C after an A failure.
8. Run `compare --comparisons` after the repeated runs finish.
9. Recheck output agreement at concurrency, not only the B1 parity pilot.
10. Review short profiler traces separately from clean latency.
11. Extend to open-loop capacity or track-specific scale only if the measured headroom justifies that work.
12. Finish the required 04/06/07/08/09 reports with explicit missing fields.
13. Commit only this PoC's paths. Preserve the outside index hash.
14. Push the experiment branch and check its remote SHA.
15. Restore owned burners on idle GPUs after the final quiet interval.

## Current evidence limits

Internal RLT handoff parity passed 100/100 prompts for every model and A/B control pair.
Official-reference sequence agreement is Ouro 88/100, Huginn 89/100, and Dory 80/100.
Do not call these values downstream accuracy.
No full GSM8K test or HumanEval result exists for this campaign.

The original single-process NIXL probe is not a validated hardware bandwidth floor.
Its host-observed rate differs from the separate-process serving transport.
A persistent two-process probe is queued before repeated serving.
Do not extrapolate an architectural bound from the single-process probe alone.

The main serving comparison includes a matched staged A control.
It also includes a generic descriptor-coalesced B control for Ouro.
The longer output-cap-128 regime prevents a cap-32-only conclusion.
All final performance claims still require three independent starts and output agreement.

## Safety

The preserved outside-index hash is `c162445edb25097abbf18627e41ba8dffc2a15f9a35e6d11385b37a9eb449a22`.
Use the exclusions in the prior session summary when checking that hash.
Never reset or clean the repository.
Never signal a process without checking its ownership and this experiment's ledger.
