# Offline main table draft: quality and work

This is a stopped, partial draft. All available rows use `inclusionAI/LLaDA2.0-mini@dad945cac317da394b390f82c7b40691d8a881ed`, BF16, threshold 0.95, block 32, identical GSM8K IDs 0–31, prompt/parser, temperature 0, and generation cap 2048.

| Method | GSM8K | HumanEval | Mean NFE | AvgK | Fresh expert-token pairs | Unique experts/forward | W_active | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Vanilla | 31/32 (96.875%) | reference only: 143/164 | 98.250 | 8 | 788,011,776 | 221.539 | 49,906 | 32 EOS |
| TEAM-PORT | N/A (not run) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| REFLEX | N/A (official code unavailable) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| OURS-F0 | 31/32 (96.875%) | N/A | 92.938 | 8 | 723,984,512 | 220.975 | 36,709 | 32 EOS |
| OURS-F1 | 29/32 (90.625%) | NOT RUN | 95.719 | 8 | 747,224,248 | 221.097 | 38,261 | 32 EOS |

Quality is `MEASURED_QUALITY`; route/work counters are measured from the actual dense-emulation trajectories. Dense-emulation wall time is excluded. The table is not ready for a publication claim because F1 failed before the n=128 gate and TEAM/REFLEX have no common-protocol rollout.

