# Q2 — Visual contribution to prefill EP communication

The direct byte view uses actual mixed-prefill routes. Physical replays use frozen full-route expert IDs and owner ranks; visual-only plus text-only latency is deliberately not added to reconstruct the nonlinear full collective.

| Size | Visual tokens (prior processor audit) | Visual token share | Visual dispatch-byte share | Full dispatch | Visual-only dispatch | Text-only dispatch | Full combine | Visual-only combine | Text-only combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 126 | 71.591% | 71.593% | 1.3469 ms | 1.4707 ms | 1.5857 ms | 0.1710 ms | 0.3121 ms | 0.3107 ms |
| medium | 672 | 93.075% | 93.031% | 1.9787 ms | 2.0722 ms | 1.7753 ms | 0.2401 ms | 0.2769 ms | 0.3562 ms |
| high | 1980 | 97.537% | 97.540% | 3.4309 ms | 3.8790 ms | 2.8824 ms | 0.1893 ms | 0.1765 ms | 0.2208 ms |

Verdict: **VISUAL_PREFILL_COMMUNICATION_DOMINANT**.

Bytes are logical BF16 activation payload after token-destination dedup; metadata/control bytes are not mislabeled as measured transport bytes. Dispatch/combine latencies are physical CUDA-event measurements.

## Per-rank mixed-route traffic

The following values sum 48 layers and all source shards for measured replicate 0.

| Size | Modality | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---|---:|---:|---:|---:|
| low | visual | 22147072 | 23265280 | 22163456 | 22470656 |
| low | text | 8331264 | 8359936 | 8482816 | 8192000 |
| low | special_other | 749568 | 532480 | 520192 | 561152 |
| medium | visual | 119992320 | 118882304 | 120832000 | 118169600 |
| medium | text | 7970816 | 7929856 | 8220672 | 7798784 |
| medium | special_other | 1126400 | 925696 | 897024 | 929792 |
| high | visual | 359014400 | 357711872 | 351670272 | 341573632 |
| high | text | 8069120 | 7733248 | 8187904 | 7737344 |
| high | special_other | 1122304 | 913408 | 872448 | 921600 |
