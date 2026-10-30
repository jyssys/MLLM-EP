# Actual progress ledger

## 2026-10-29, setup and qualification

The budget clock starts at21:29:30 KST, before setup and source inspection.
The full223-line authoritative specification was read.
The new branch starts at the frozen parent commit. The previous A/B/C code remains unchanged.
GPU0–7 UUIDs and NV18 topology were checked. Ownership validation stopped only approved burner sessions.

The pinned ShareGPT source produced24 calibration,40 correctness,200 W1,100 W2,100 W3, and80 W4 prompts.
The three primary subsets are disjoint by prompt hash.
W1 prompt-token P50/P90 is715.5/1784.2 for Ouro and685/1753.3 for Huginn.
These values describe the selected text population. They are not a production traffic distribution.

The four paired correctness jobs used all eight GPUs concurrently.
Both models passed40/40 exact outputs and depth traces at cap32.
The separate two-request KV/hidden jobs passed byte identity, including warmup and reversed fragments.
Fresh official-HF triage gives Ouro7/8 and Huginn6/8 sequence agreement.
The discrepancy remains unresolved. These counts are not answer accuracy.

## Baseline calibration and held-out start

Two isolated low-load calibration arms per model completed on GPU0,1.
The frozen primary SLO is TTFT857ms and TPOT30ms for Ouro.
The frozen primary SLO is TTFT1454ms and TPOT119ms for Huginn.
The secondary observed-ITL thresholds are31ms and121ms. They do not select the primary winner.
The formula and calibration samples remain in `configs/slo_lock.json`.

The first Ouro W1 DP_Mixed run completes200 arrivals at1 request/s.
Its joint point attainment is83%. This is not an accepted capacity estimate.
The controller continues matched PD_G1/B and both-model rate sweeps without changing the SLO.
The prospective longer-window refinement permits lower-rate feasibility checks when the initial baseline ladder has no passing point.
That refinement uses baseline feasibility, not a favorable B E2E result.

The private RLT model/runtime patch remains SHA-256 `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
The source freeze confirms identical held-out worker source for the first DP_Mixed and PD_G1 runs.
A CPU-only waiting supervisor was replaced once to strengthen orphan-worker protection.
No inference worker received a signal during that watcher replacement.

Later progress is authoritative in `state/campaign.json`, `commands.jsonl`, and per-run status files.
This ledger does not claim that the remaining experiment stages are complete.

## 2026-10-29, first held-out serving gate

Ouro's first W1 B/PD_G1 comparison gives185/200 identical output sequences.
All200 inputs match. Output lengths and depth-trace lengths match198/200.
Every recorded depth value remains4. The model does not use early exit.
The controller stops subsequent Ouro B performance runs. Existing timings remain output-confounded.
Huginn's first W1 DP_Mixed run continues in the isolated timing lane.

The EOS analysis now counts the pinned scheduler's `STOP` reason as EOS.
This changes report labels only. The worker source and model execution stay frozen.
Five CPU contract tests pass after this analysis change.

The CPU-only waiting supervisor was replaced again to add post-serving mismatch diagnostics.
Its current PID is3394636. No inference worker received a signal.
The planned diagnostics compare a repeated PD_G1 control and sequential mismatching prompts.
They cannot reopen the failed B serving-performance gate.

## 2026-10-29, 23:08 KST, Huginn held-out gate

Huginn's first W1 B/PD_G1 comparison gives193/200 identical output sequences.
All200 inputs match. Output lengths and recorded depth traces match198/200.
Every recorded output depth remains32.
The controller disables further Huginn B serving-performance runs.
No model now has an open B performance-acceptance gate.

At nominal0.2requests/s, the raw Huginn joint point attainment is93.5% for DP_Mixed,84.5% for PD_G1, and81.0% for B.
The B row remains output-confounded. It is not an equivalent-work performance result.
The three runs use200 planned arrivals and the same frozen primary SLO.
Baseline comparison, independent starts, supplementary workloads, and correctness diagnostics continue within the original budget.

## 2026-10-30, 02:38 KST, B-priority repair

The user requested cause diagnosis and repair before further B serving measurements.
CPU campaign3344105 is paused. Its healthy in-flight Huginn job finished normally.
Repair controller3568550 owns the quiet lane. Qualification watcher3630273 expands the canonical candidate.
Supervisor3394636 still enforces the original deadline. The15-hour clock is unchanged.

The actual causal-prefix regression fails for both B/control and an identical-control repeat.
Ouro gives7/8 and7/8 exact sequences. Huginn gives31/33 and30/33.
The late-only Ouro capture reproduces the original request3 mismatch at output index125.
Its logits differ by at most0.125. A BF16 top-logit tie changes greedy argmax from4533 to30.

The target R3 prompt KV matches exactly across the original pair.
Its earlier generated history also matches. Its prior decode KV differs in551,940 elements.
The canonical linear plus fixed-capacity core candidate gives0 KV differences and8/8 output parity in the same small diagnostic.
This is not a full repair acceptance or a measured serving speedup.

The qualification ladder tests Ouro8/32/200 and Huginn33/200 requests.
It includes a repeated control at every stage.
Only models with full qualification, fixed-depth work, stop parity, source hashes, and byte checks enter a new serving epoch.
The original failure files and SLOs remain unchanged.
Post-repair results use separate files and never pool with original timings.
Nine CPU contract tests pass. No final commit or push is claimed yet.
