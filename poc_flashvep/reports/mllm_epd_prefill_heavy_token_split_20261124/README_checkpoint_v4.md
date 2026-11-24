# MLLM EPD checkpoint v4

Status: ACTIVE and INCOMPLETE. Main100 remains active. Main400 remains unchanged.
The low-load QPS0.5 CUPTI clean/profile/clean test passed all five frozen5% observer checks.
Central-window E/P/D kernel busy was12.7326%/14.0682%/14.3716%.
Simultaneous E/D physical kernel idle covered74.2376% of that window.
This was low load, not a demonstrated prefill-heavy regime or safe extra-prefill budget.
The shortened QPS3 capture failed observer checks. It remains DIAGNOSTIC.
E/D prefix quantum cost controls also failed their whole-arm observer gates.
Those costs cannot establish qualified safe quanta or an online serving improvement.
The128-token probe imported a complete2691-row image embedding,19289088 bytes.
That observation concerns image-level transport granularity. It is not visual-token compute complexity.
A new common compiled BI0 policy passed15 serial correctness cases in each of two independent initializations.
The original stock compiled strict failures remain unchanged.
The new policy controls initial reduction candidates and disables incompatible fusion benchmarking.
Original BF16 weights, model forward, attention, vision and processor remain unchanged.
The separate concurrent N8 control passed all28 imported-prefix KV hashes and query MRoPE checks.
Actual P and D batches reached3, not8. Each request generated only one first output.
This does not establish full suffix cross-batch equivalence, task quality or active-decode safety.
Task quality remains PENDING. Token C3 serving speedup and safe headroom remain unproved.
CPU regression:124PASS. The earlier failing CPU control log is preserved.
Common-policy Native C32 is live and excluded from these evidence records.
No final goal completion or GPU closeout is claimed. Approved burners cover unused GPUs.
