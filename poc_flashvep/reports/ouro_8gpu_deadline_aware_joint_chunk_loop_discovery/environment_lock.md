# Environment lock

The source inventory is `source_locks/lock_700606e0.json`.
The initial physical inventory and UUID map are in `registration_lock.json`.
Each worker checks its loaded Torch device UUID in `raw/JOB/identity.json`.

| Component | Recorded value |
| --- | --- |
| GPU | Eight NVIDIA H100 80GB HBM3 devices |
| Driver | 570.211.01 |
| Interconnect | NV18 between all GPU pairs |
| Host | Intel Xeon Platinum 8480+, 192 visible CPU IDs, one exposed NUMA node |
| Python | 3.14.6 |
| Torch | 2.11.0+cu129 |
| Attention | FlashAttention 4, 4.0.0b30 |
| Triton | 3.6.0 |
| Transformers | 5.16.1 |
| NumPy | 2.4.6 |
| Model | Base Ouro-1.4B, revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 |
| RLT | 4a350344027960a48384626aa646f1765fda6d4f |
| Candidate diff SHA256 | 1feb47c40a10e2f6c0cc9a314aec86ff96c6b251954eaed31d1be0b859f67862 |

The Python path is `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python`.
The runtime checkout is `/home/esjung/external/ouro-deadline-g3-candidate-rlt`.
The original calibration and previous-study checkouts remain intact.

Each process receives exactly one physical GPU through `CUDA_VISIBLE_DEVICES`.
The worker rejects an unexpected UUID or an initialized distributed process group.
Each GPU has separate compilation caches.
All six policy cells remain inside one GPU/process-start block.
`gpu_assignment_history.csv` records preferred and actual device assignment.

Each main engine reserves 18,432 KV pages of 16 tokens across its depth storage.
The KV pool occupies 54 GiB.
That allocation is not measured HBM read traffic.
Unique model parameter and buffer storage is 2,869,305,602 bytes.
The model has 24 physical layers and 96 logical layer/depth KV namespaces.

Only recognized experiment-owned burners receive stop signals.
Finished devices receive restored burners.
Quiet controls reserve otherwise idle devices without burners.
The resource helper supplies an explicit GPU list because the shell script has an outdated default mask.
