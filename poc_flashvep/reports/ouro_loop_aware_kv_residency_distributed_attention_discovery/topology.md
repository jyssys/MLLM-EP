# Physical topology

`topology.txt` contains the actual `nvidia-smi topo -m` output.
Every GPU pair reports NV18.
The preferred operator groups are physical GPUs 0–1, 0–3, and 0–7.
All devices report NUMA node 0 and CPU affinity 0–191 in that output.
This output does not measure achieved bandwidth.

Each distributed rank records actual peer-access support.
The two-rank transfer test measures NCCL P2P with completion barriers.
It reports bytes, latency, and effective bandwidth including synchronization.
That bandwidth is not raw link throughput.
The operator measures real query broadcast and partial-result all-gather.
Its full latency includes merge operations and communication.

Single-GPU workers use separate CUDA contexts and CPU-affinity ranges.
The registered 1/2/4/8-worker calibration uses identical workload shapes.
The isolation groups leave unused devices idle by reservation.
Parallel discovery runs and quiet comparisons remain separate evidence classes.
