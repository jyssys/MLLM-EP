"""Own and physically touch a persistent EPD device arena until the lease ends."""
import argparse
import json
from pathlib import Path
import signal
import time

GIB = 2 ** 30
POST_WARM_DEVICE_FREE_MARGIN_GIB = 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-gib", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--role", choices=("encoder", "prefill"), default="encoder")
    args = parser.parse_args()
    if args.reserve_gib <= 0:
        parser.error("A positive physical allocation is required")

    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Reserve worker must have exactly the leased {args.role} GPU mapped")
    free_before, total = torch.cuda.mem_get_info()
    if free_before < (args.reserve_gib + POST_WARM_DEVICE_FREE_MARGIN_GIB) * GIB:
        raise RuntimeError("The arena requires 4 GiB of additional physical free memory after the stock warm cohorts")
    arena = torch.empty(args.reserve_gib * GIB, device="cuda:0", dtype=torch.uint8)
    arena.zero_()
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    value = {
        "size_bytes": arena.numel() * arena.element_size(), "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(), "free_before_bytes": free_before,
        "free_after_bytes": free_after, "device_total_bytes": total, "ready_monotonic": time.monotonic(),
        "physical_free_margin_gib": POST_WARM_DEVICE_FREE_MARGIN_GIB,
        "role": args.role,
        "device": str(torch.cuda.get_device_properties(0)),
        "scope": f"Separate CUDA allocator on the same {args.role} GPU. Arena touched before measurement and held "
                 "without ongoing kernels. Additional assist workspace and latency neutrality are not qualified.",
    }
    with args.receipt.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
