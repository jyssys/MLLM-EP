# Existing execution options before successor claims

Controls include eager versus graph; unsplit versus one-part multi-stream;
two/four FFN parts; identical-plan independent restarts; same-prefix independent
HF reference; full-SM versus native 112/16 green-context settings; and warmed
large-prefill versus decode cohorts. The resource counts are actual native
stream choices, but not proven physically disjoint category allocations.

Graph capture removes much of the eager launch overhead: this is an existing
runtime option, not successor novelty. The extra plan/search opportunity must
be measured against the graph-enabled appropriate baseline. Cold graph setup
is charged in decode request E2E and separated from steady ITL; treating that
one-time cost as repeated online waste would be incorrect.

Pure-prefill comparison has no decode-capture confound: eager plans initialize
in two exact-shape warmup cohorts. Plain beats the split plans' median request
costs. Additional selection headroom is zero in that finite set.

The obvious last-token projection repair and completing missing MoE profile
hooks are baseline engineering, not a new research method. We do not claim a
percentage of original-paper speedup recovered by a manually chosen plan.
