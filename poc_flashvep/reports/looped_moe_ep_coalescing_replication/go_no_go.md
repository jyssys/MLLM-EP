# Decision record

Overall implementation decision: **NO_GO_EP_DIRECTION for the tested A/B policy families.**
Production extrapolation: **CONDITIONAL**, not a universal negative result.

| Model | EP | P0 TRUE EP | P1 characterization | P2 models | A | B | P3 method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | PASS | PASS, bounded | CONDITIONAL O2 | NO_GO_A | NO_GO_B | NOT_RUN |
| IFM Ouro | 8 | PASS | PASS, bounded | CONDITIONAL O2 | NO_GO_A | NO_GO_B | NOT_RUN |
| IFM Huginn | 4 | PASS | PASS, bounded | CONDITIONAL O2 | NO_GO_A | NO_GO_B | NOT_RUN |
| IFM Huginn | 8 | PASS | PASS, bounded | CONDITIONAL O2 | NO_GO_A | NO_GO_B | NOT_RUN |
| Foil-1 | 4 | PASS | PASS, bounded | CONDITIONAL O2 | NO_GO_A | NO_GO_B | NOT_RUN |
| Foil-1 | 8 | PASS | PASS, bounded | CONDITIONAL O2 | NO_GO_A | NO_GO_B | NOT_RUN |

| Model | EP | A vs best generic | B vs static, max | B vs generic, max | B load-only ideal |
| --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 1.000× | 1.068× | 1.002× | 1.083× |
| IFM Ouro | 8 | 1.000× | 1.094× | 1.002× | 1.112× |
| IFM Huginn | 4 | 1.000× | 1.055× | 1.004× | 1.074× |
| IFM Huginn | 8 | 1.000× | 1.067× | 1.007× | 1.092× |
| Foil-1 | 4 | 1.000× | 1.006× | 1.001× | 1.016× |
| Foil-1 | 8 | 1.000× | 1.009× | 1.008× | 1.022× |

All headroom ratios are modeled, not measured optimized E2E speedups.
Neither track reaches 1.15× against its strong matched generic control.
Replication also fails the 5% incremental recurrence criterion.
The evidence supports stopping these prototypes, not declaring all looped EP research impossible.

B1 cannot coalesce independent requests because only one request exists.
B2 shows no cross-pass ready pairs in this diagnostic.
B4 Foil shows pairs, but bounded waiting does not beat tuned generic batching.
Loop-conditioned placement changes some sets but adds less than 1% over generic sets.
Actual optimized A/B accuracy and E2E are NOT_RUN.
DeepEP, production paged-KV serving, and a globally optimal scheduling bound are NOT_RUN.
