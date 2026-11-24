# MLLM EPD checkpoint v2

Status: ACTIVE and INCOMPLETE. Earlier reports and checkpoints remain unchanged.
Main100 is active. Main400 remains available without modification.
Agg DP3, E+PD DP2 and Native EPD completed the common three-GPU Natural baseline sweep.
QPS2 and QPS4 use three independent server starts. Other QPS points use one discovery start.
The shared QPS grid is0.5,1,2,3,4,6. Actual points and start-level intervals have corresponding CSV files.
Native QPS2 reuses the qualified C11 runtime records. It does not repeat them under a new label.
Full task quality remains PENDING. Runtime timing is not a task-quality pass.
The official Mantis scorer retains its48% result for E+PD S2, versus52% for the reference.
A diagnostic shows that the single different score concerns two responses with the same final option A.
The official parser does not recognize one response format. No answer or official score was changed.
C15 failed before GPU loading because its preflight counted source receipts as GPU ownership leases.
The preserved red test and corrected lease check distinguish source receipts from actual server leases.
C15R1 ran the first exact token boundary inside an image span.
All28 imported prefix KV layers were byte-exact. Prompt, MRoPE and prepared embeddings matched.
All16 output tokens matched the chunk-aligned teacher. Full suffix KV strict equality failed.
No token performance run followed this failed gate. Separate numerical diagnosis continues.
C16 loaded an extra original full model beside the encoder, with about8GiB of KV storage.
Three predefined real input stresses completed. All68 encoder features matched before and after loading.
This proves observed memory fit only. Extra prefill safety and maximum encoder batch capacity remain unmeasured.
CUPTI observer failures remain diagnostic. No decode-safe headroom or token-split speedup is claimed.
Approved burners remain on unused GPUs. This checkpoint is not final resource closeout.

Main100 SHA256:910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1.
Model/processor revision:cc594898137f460bfe9f0759e9844b3ce807cfb5.
CPU regression result:85PASS. GPU correctness and task-quality gates remain separate.
Closed raw subset:archives/checkpoint_raw_records_v2.json.
The final campaign must verify the remaining gated scope, archive evidence and restore all approved burners.
