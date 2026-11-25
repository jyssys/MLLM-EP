"""Start kernel-only CUPTI near the fixed central window, then flush after drain."""
import asyncio
import time

from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import rpc
from ..mllm_epd_h200_prefill_heavy_batch_matched.runtime import ROLES


def short_roi_hook(name, state_interval_ms, decode_buffer_gib):
    records = {}

    async def capture(start, trace, measurement_done):
        ports = {role: port for role, _, port, _ in ROLES}
        left = start + trace[int(len(trace) * .2)]["offset_seconds"]
        await asyncio.sleep(max(0, left - 3.0 - time.monotonic()))

        async def begin(role, port):
            result = await asyncio.to_thread(rpc, port, "profile/begin", {
                "name": name, "include_copies": False,
                "state_interval_ms": state_interval_ms,
                "buffer_gib": decode_buffer_gib if role == "decode" else 1,
            })
            records[role] = {"begin": result}

        try:
            begun = await asyncio.gather(*(begin(role, port) for role, port in ports.items()),
                                         return_exceptions=True)
            errors = [exc for exc in begun if isinstance(exc, BaseException)]
            if errors:
                raise errors[0]
            await measurement_done.wait()
        finally:
            roles = list(records)
            finished = await asyncio.gather(*(
                asyncio.to_thread(rpc, ports[role], "profile/end", {}) for role in roles
            ), return_exceptions=True)
            for role, value in zip(roles, finished, strict=True):
                if isinstance(value, BaseException):
                    records[role]["close_error"] = str(value)
                else:
                    records[role]["end"] = value

    return capture, records
