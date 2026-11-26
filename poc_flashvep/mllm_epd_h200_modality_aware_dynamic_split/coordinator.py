"""The P boundary callback and its request-scoped D ownership transaction."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

import aiohttp

from .late_worker import RPC_TIMEOUT_SECONDS
from .policy import BLOCK, candidate_offsets


@dataclass
class Prefill:
    length: int
    cuts: list
    spans: list
    required_blocks: int
    E_http_ms: float
    ratio_bands: list
    force_continue: bool = False
    decisions: list = field(default_factory=list)
    image_keys: tuple = ()

    def supports(self, prefix):
        if not 0 < prefix < self.length:
            return False
        ratio = (self.length - prefix) / self.length
        return prefix in self.cuts or any(low <= ratio <= high for low, high in self.ratio_bands)


class LateCoordinator:
    def __init__(self, admission, policy, decoder, metadata_root, *, require_ready_transport=False):
        self.admission = admission
        self.policy = policy
        self.decoder = decoder
        self.metadata_root = Path(metadata_root)
        self.prefills = {}
        self.require_ready_transport = require_ready_transport

    def register(self, request_id, length, max_output, trace):
        key = hashlib.sha256(request_id.encode()).hexdigest()
        record = json.loads((self.metadata_root / key / "manifest.json").read_text())
        if record["prompt_tokens"] != length:
            raise RuntimeError("E processor geometry mismatch")
        spans = [tuple(span) for span in record["image_spans"]]
        if len(spans) != record["images"] or any(not 0 <= start < end <= length for start, end in spans):
            raise RuntimeError("Invalid E visual spans")
        forced = trace.get("late_correctness_cut")
        if forced is not None and not 0 < forced < length:
            raise ValueError("Correctness target outside full input")
        natural = self.policy.natural or trace.get("late_correctness_natural", False)
        hold = self.policy.natural_hold or trace.get("late_correctness_natural_hold", False)
        if natural:
            if forced is not None:
                bands = [((length - forced) / length,) * 2]
            elif hasattr(self.policy, "candidate_offsets"):
                bands = self.policy.natural_bands(length)
            else:
                bands = self.policy.natural_bands()
            cuts = []
        else:
            bands = []
            if forced is not None:
                cuts = [forced]
            elif hasattr(self.policy, "candidate_offsets"):
                cuts = self.policy.candidate_offsets(length)
            else:
                cuts = candidate_offsets(length, spans, self.policy.mode, self.policy.ratios)
        item = Prefill(length, cuts, spans, max(256, (length + max_output + BLOCK - 1) // BLOCK),
                       (trace["E_complete"] - trace["E_submit"]) * 1000,
                       bands,
                       force_continue=trace.get("late_correctness_continue", False),
                       image_keys=tuple(record.get("image_hashes", ())))
        if request_id in self.prefills:
            raise RuntimeError("Duplicate P decision owner")
        self.prefills[request_id] = item
        trace["late_plan"] = {"policy": self.policy.mode, "length": length, "candidate_cuts": cuts,
                              "natural": natural, "natural_hold": hold, "ratio_bands": bands,
                              "shadow": self.policy.shadow, "image_hashes": record.get("image_hashes", []),
                              "image_spans": spans, "required_KV_blocks": item.required_blocks,
                              "decisions": item.decisions}
        plan = {"request_id": request_id, "cuts": cuts}
        if natural:
            plan.update(natural=True, ratio_bands=bands)
            if hold:
                plan["hold"] = True
        return plan

    def ready_suffix_transport(self, item, prefix):
        from ..mllm_epd_prefill_heavy_token_split.crossing_ipc import decode_inline_roles, inline_available

        if len(item.image_keys) != len(item.spans):
            return False, 0, None
        required = [key for (_, end), key in zip(item.spans, item.image_keys, strict=True) if end > prefix]
        missing = 0
        for key in required:
            folder = self.metadata_root.parent / key
            if (not (folder / "cuda_ipc_ready.json").is_file()
                    or not any(inline_available(folder, role) for role in decode_inline_roles(folder, True))):
                missing += 1
        return missing == 0, len(required), missing

    async def expire(self, request_id):
        self.prefills.pop(request_id, None)
        return await self.admission.release_unclaimed(request_id)

    async def decide(self, body, session):
        begin = time.monotonic()
        lease = body.get("lease", {})
        request_id = lease.get("request_id")
        prefix = lease.get("prefix_tokens")
        item = self.prefills.get(request_id)
        if (item is None or type(prefix) is not int or not item.supports(prefix)
                or body.get("full_prompt_tokens") != item.length):
            return {"accepted": False, "reason": "NO_MATCHING_P_BOUNDARY"}

        note = {"prefix_tokens": prefix, "full_prompt_tokens": item.length, "callback_begin": begin,
                "P_boundary_state": body, "D_before": self.admission.state()}

        def finish(accepted, reason):
            note.update(accepted=accepted, reason=reason, callback_end=time.monotonic())
            item.decisions.append(note)
            return {"accepted": accepted, "reason": reason}

        if not self.admission.idle_for_reservation():
            return finish(False, "D_ADMISSION_BUSY")
        if item.force_continue:
            return finish(False, "CORRECTNESS_CONTINUATION_CONTROL")
        if self.require_ready_transport:
            ready, required_images, missing = self.ready_suffix_transport(item, prefix)
            note.update(transport_required_suffix_images=required_images,
                        transport_missing_inline_images=missing, transport_ready_at_cut=ready)
            if not ready:
                return finish(False, "SUFFIX_INLINE_TRANSPORT_UNAVAILABLE")
        policy_keywords = {"progress": body.get("progress")} if hasattr(self.policy, "candidate_offsets") else {}
        accepted, reason, policy_data = self.policy.choose(
            length=item.length, prefix=prefix, spans=item.spans, cuts=item.cuts,
            competitors=body.get("competitors", []), state_observed=body.get("competitors_observed") is True,
            E_http_ms=item.E_http_ms, **policy_keywords)
        note.update(policy_data)
        if not accepted:
            return finish(False, reason)
        boundary_ready = body.get("boundary_ready")
        if not isinstance(boundary_ready, (float, int)):
            return finish(False, "P_BOUNDARY_CLOCK_MISSING")
        # Leave a small local response allowance before the P worker's bounded read.
        deadline = boundary_ready + RPC_TIMEOUT_SECONDS - .002
        time_left = deadline - time.monotonic()
        if time_left <= 0:
            return finish(False, "P_CALLBACK_EXPIRED")
        try:
            async with session.post(self.decoder + "/campaign/runtime_state", json={},
                                    timeout=aiohttp.ClientTimeout(total=min(.015, time_left))) as response:
                response.raise_for_status()
                worker, = await response.json()
            engine, = worker["schedulers"]
        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError):
            return finish(False, "D_ENGINE_STATE_UNAVAILABLE")
        note["D_engine"] = engine
        if (engine["running"] or engine["waiting"] or engine.get("skipped_waiting", 0)
                or engine.get("unfinished", 0)):
            return finish(False, "D_ENGINE_BUSY")
        if engine["KV_free_blocks"] < item.required_blocks:
            return finish(False, "INSUFFICIENT_D_KV")
        if request_id not in self.prefills:
            return finish(False, "P_ALREADY_RETURNED")
        if self.policy.shadow:
            eligible = self.admission.idle_for_reservation() and time.monotonic() < deadline
            note["shadow_would_reserve"] = eligible
            return finish(False, "SHADOW_ELIGIBLE" if eligible else "SHADOW_D_CHANGED")
        reservation = await self.admission.reserve(request_id, prefix, engine, deadline=deadline)
        if reservation is None:
            return finish(False, "D_RESERVATION_RACE")
        note["reservation"] = reservation
        return finish(True, reason)
