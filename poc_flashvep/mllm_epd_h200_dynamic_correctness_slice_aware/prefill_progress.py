"""Cheap causally available P work features shared by observations and decisions."""
from heapq import nsmallest
from itertools import chain


def candidates(scheduler, exclude, retiring):
    """Describe the nearest other requests using rows retired or issued so far."""
    def record(request, waiting):
        length = request.num_prompt_tokens
        issued = min(length, request.num_computed_tokens)
        inflight = max(0, request.num_in_flight_tokens - retiring.get(request.request_id, 0))
        retired = issued - inflight
        return {
            "prompt_tokens": length, "issued_prompt_tokens": issued,
            "retired_prompt_tokens": max(0, retired), "inflight_prompt_tokens": inflight,
            "remaining_prompt_tokens": max(0, length - issued), "waiting": waiting,
        }

    work = (record(request, waiting) for request, waiting in chain(
        ((req, False) for req in scheduler.running),
        ((req, True) for req in chain(scheduler.waiting, scheduler.skipped_waiting)))
        if request.request_id != exclude)
    return nsmallest(8, work, key=lambda row: row["remaining_prompt_tokens"])
