
# Q1 token paths versus active experts

Actual installed BF16 Triton grouped-expert kernel on physical GPU2, d=2048/m=768. Each cold observation uses a 256 MiB L2-displacing preconditioner. Warmup=12, repeats=40, rounds=3. This is rank-local kernel evidence, not full EP latency.

| Local expert bank | 2 active ms | 4 active ms | 8 active ms | 16 active ms | 2→16 effect | joint R² increment over token-only |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.05881 | 0.06375 | 0.06144 | 0.08794 | 49.53% | 0.853 |
| 32 | 0.05828 | 0.06085 | 0.06470 | 0.08877 | 52.33% | 0.232 |
| 64 | 0.05616 | 0.06005 | 0.06658 | 0.09263 | 64.95% | 0.753 |

At fixed 32 paths, the mean cold 2→16 active-expert effect is **55.60%**. The mean joint-model R² increment over token-only is **0.613**. Hot repeated-weight results are retained in the artifact and are not used to infer cold HBM behavior.
