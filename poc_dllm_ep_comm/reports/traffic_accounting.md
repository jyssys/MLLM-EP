## 4. Actual EP bytes per final output token

Actual full-request logical remote D+C payload: 4.679 MiB/token (request-bootstrap95% CI 4.003–5.351 MiB/token).
Output-position-only payload: 3.885 MiB/token (request-bootstrap95% CI 3.254–4.522 MiB/token).
These are **backend-validated logical useful payload bytes**, not NVLink packet
or NIC wire counters. A BF16 hidden/output row is4096bytes. DeepEP normal sends
one row per distinct destination rank even if several selected experts share
that rank. Combine returns one partial-output row per corresponding destination.
Local expert paths are separate and contribute0 remote payload bytes. Router
metadata, protocol bytes, shared-expert compute and dense TP collectives are not
included in this sparse-EP byte numerator. They remain in measured request wall.

## 5–8. Actual vs live-only vs acceptance-route one-shot

For each actual source position/layer call, route owners come from loaded
ownership. `B_actual` counts all executed remote payloads. `B_live_only` retains
only output positions that were masks before that call, even if not accepted
afterwards. `B_one_shot` sums each returned output's first-acceptance routing
exactly once in each19 sparse layers. Prompt/known-context is not called
already-decoded output.

`B_actual = B_context + B_dead_output + B_live_only` is asserted exactly.
Already-accepted output accounts for 49.153% (request-bootstrap95% CI 47.298–50.975%) of actual
output-position traffic, or 56.76% of output repetition above one-shot.
The remaining output excess is repeated live work (plus eventual EOS/tail where
present). 75.093% (request-bootstrap95% CI 69.206–80.756%) of live-only remote bytes belongs
to positions still unaccepted after that iteration. This is acceptance-inefficient
traffic, not demonstrably safe-to-remove waste.

![Actual decomposition](figures/main_v2/actual_traffic_decomposition.png)

