# MLLM EPD checkpoint v1

Status: ACTIVE and INCOMPLETE. Earlier checkpoint documents remain unchanged.
Main100 is the active workload. Main400 and original prompts are preserved.
Native EPD completed three independent starts at QPS2, with Natural EOS and equal-output replay kept separate.
Natural start-mean TTFTp95 is4880.240ms, ITLp99 is16.267ms, TPOTp95 is6.827ms, and Joint SLO is91.667%.
These values do not establish a token-split speedup or complete task quality.
The official MMLongBench-Doc extraction pipeline remains unavailable without an authorized external API.
Native CPU/IPC and publication/ACK interventions preserve their negative results and scopes.
C11S2 and C13 GPU profiles failed the frozen5% observer gate. Their kernel/idle results stay diagnostic.
C12 failed during initialization because identical source archives shared an extraction directory.
The failure was preserved. A red-capable regression test now covers the per-boot identity fix.
Future server processes execute verified extracted source snapshots.
Generic token ownership, exact remote prefix counting, original multimodal metadata and independent image-consumer plans have CPU tests.
Generic token-split GPU correctness, active-decode insertion, and online serving gains remain NOT_RUN.
The new stock Agg DP3/E+PD DP2/Native three-GPU Natural sweep is running separately asC14.
No Source Credit, D Cap or original model forward was weakened for that comparison.

## Evidence

- Main100 manifest SHA256:910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1.
- Model/processor revision:cc594898137f460bfe9f0759e9844b3ce807cfb5.
- Native table:analysis/native_inline_independent_N3_v0.csv and matchingJSON.
- CPU tests:71PASS. GPU model correctness and task scoring are separate gates.
- archives/checkpoint_raw_records_v1.json verifies the closed Native raw subset losslessly.
- Nested GPU traces,tensors,media and active baseline logs remain local at this checkpoint.
- No final resource closeout is claimed. Approved burners remain on unused GPUs.
