"""Share Native admission and reserve an idle D without queuing a suffix."""
import time

from ..mllm_epd_prefill_heavy_token_split.idle_admission import IdleAdmission


class LateAdmission(IdleAdmission):
    def __init__(self, decode_cap):
        super().__init__(decode_cap=decode_cap)
        self.reservation = None

    def idle_for_reservation(self):
        return not (self.failure or self.active or self.pending or self.prefill_owner)

    async def reserve(self, request_id, prefix, engine_state, deadline=None):
        async with self.condition:
            if not self.idle_for_reservation() or deadline is not None and time.monotonic() >= deadline:
                return None
            before = self.state()
            self.prefill_owner = request_id
            self.reservation = {"request_id": request_id, "prefix_tokens": prefix,
                                "reserved": time.monotonic(), "before": before,
                                "engine_state": engine_state}
            return self.reservation

    async def claim(self, request_id, prefix):
        async with self.condition:
            reservation = self.reservation
            if (reservation is None or reservation["request_id"] != request_id
                    or reservation["prefix_tokens"] != prefix or self.prefill_owner != request_id
                    or self.active or self.failure):
                raise RuntimeError("D suffix requires its own idle reservation")
            before = self.state()
            now = time.monotonic()
            self.active[request_id] = True
            self.reservation = None
            event = {"request_id": request_id, "extra_prefill": True, "begin": now,
                     "admitted": now, "enqueued": before, "before": before, "after": self.state(),
                     "suffix_isolated": True, "blocked_on_enqueue": [], "reservation": reservation}
            self.events.append(event)
            return event

    async def release_unclaimed(self, request_id):
        async with self.condition:
            if self.reservation is None or self.reservation["request_id"] != request_id:
                return False
            assert self.prefill_owner == request_id and not self.active
            self.reservation = None
            self.prefill_owner = None
            self.condition.notify_all()
            return True
