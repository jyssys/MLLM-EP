# TRUE EP4 path and ownership proof

Status: **PASS for physical sparse EP4 execution**

## Actual process groups

- Physical devices: GPU 4, 5, 6, 7, four distinct H100 UUIDs in one NV18
  clique.
- TP=2, DP=2, SP=1, PP=1, EP=4. EP is the combined TP×DP expert group; these
  sizes are not multiplied into sixteen devices.
- Backend: DeepEP high-throughput dispatch/combine plus sparse Triton expert
  kernels.
- Linear owner map: rank 0 experts 0–31, rank 1 32–63, rank 2 64–95, rank 3
  96–127.

Runtime logs instantiate `DeepEPHTAll2AllManager`, filter each rank to 32 of
128 expert weights, and invoke the DeepEP prepare/finalize path. The sparse
kernel receives the selected top-8 assignments; this is neither four
independent replicas nor TP-only all-reduce.

## Traffic evidence

The preserved route anchor contains 24 selected distinct requests × 48 routed
layers (1,152 layer/request rows):

- 3,016,704 logical expert assignments and 377,088 logical rows;
- 1,030,241 remote unique token-destination rows in EP4 replay;
- 4,219,867,136 logical dispatch bytes and 9,272,221,696 logical combine
  bytes under the recorded backend contract;
- maximum source remote fanout 3;
- rank assignment totals `[748911, 768039, 736420, 763334]`.

Non-zero source/destination traffic, remote owner execution, and combine were
also observed in CUDA-event profiling. TP copies, physical padding, and
logical rows remain distinct fields in the trace schema.

## Correctness boundary

The path proof is PASS. End-to-end numerical/quality parity remains UNRESOLVED
because only manual image sanity was paired and exact token agreement was 2/12.
Backend existence is not used as a quality proof.
