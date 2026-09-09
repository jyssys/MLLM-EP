# Final GPU state — 2026-09-09 15:04 KST

Research collection ended about 14:58 KST. All research worker groups exited;
`nvidia-smi -i 4,5,6,7 --query-compute-apps=...` showed no compute process before
starting burn. No experiment is queued behind the burn.

The user-authorized existing script was run with **both** overrides, because
the currently installed shell script otherwise defaults to physical 1–4:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 \
  bash /home/esjung/vllm-ep/run_utilize.sh
```

The first detached launch did not persist and left an empty local log. The
persistent-session retry is verified: supervisor PID **596994**, worker PIDs
597237/597238/597239/597240 assigned to physical 4/5/6/7, respectively. All four
printed `generation started`; NVML at 15:04:20 reports 100% utilization on all
four and about 71,549 MiB each. Engine PIDs 598090/598097/598083/598086 map to
the recorded authorized GPU UUIDs. No burn worker targets 0–3.

This is **burn only**, not continuing research. The supplied script runs for
108,000 seconds (**30 hours after model loading**) unless stopped. It is excluded
from research GPU-time/throughput/latency tables. The script was not modified.

PID values are a dated observation, not permanent process identities. Before
stopping/restarting, recheck the command, ownership and GPU mapping; do not kill
arbitrary stale/reused PIDs. The supervisor's SIGTERM handler stops its own
worker process groups.
