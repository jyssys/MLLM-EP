# Per-Shape Wave Replay

Observer-heavy traces were normalized to each mini's clean BCT, then expressed as
model milliseconds per request-forward. This deliberately gives RAWS all clean BCT
as potentially controllable time.

| refinement state | mini1 | mini2 | mini4 | mini8 | mini16 | mini32 |
|---|---:|---:|---:|---:|---:|---:|
| early | 74.623 | 36.715 | 18.844 | 10.542 | 6.029 | **5.528** |
| middle | 74.165 | 35.934 | 18.632 | 9.586 | 5.565 | **3.965** |
| late | 74.398 | 35.784 | 18.657 | 9.876 | **6.958** | 9.666 |

The late row appears to favor mini16, but this phase table is **not a valid dynamic
oracle**: mini16 late observations have a median of 16 ready requests/M512, whereas
mini32 late observations have a median of only 5.5 ready requests/M176. Selecting
mini16 from that row would invent requests that are not ready.

The corrected replay partitions each exact mini32 ready set and never changes its
request membership. Its static reconstruction error is 0.31%, 7.00%, 7.02%,
21.13%, 18.13%, and 0% for mini1 through mini32 respectively. The model is therefore
an approximate and deliberately optimistic oracle, not a substitute for clean
latency. The decisive result remains robust because it finds only one useful split.

Evidence: [`SHAPE_REPLAY_MATRIX.csv`](../SHAPE_REPLAY_MATRIX.csv),
[`STATIC_REPLAY_VALIDATION.csv`](../STATIC_REPLAY_VALIDATION.csv), and
[`PER_WAVE_ORACLE.csv`](../PER_WAVE_ORACLE.csv).
