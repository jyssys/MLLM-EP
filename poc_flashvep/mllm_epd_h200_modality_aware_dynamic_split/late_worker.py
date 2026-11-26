"""Opt-in P cut at a completed vLLM chunk. Preserve the running request on denial."""

from dataclasses import dataclass, field
from heapq import nsmallest
import http.client
from itertools import chain
import json
import os
import time
from urllib.parse import urlsplit


LATE_KEY = "campaign_late_bound_v1"
PLAN_KEY = "campaign_token_split_v1"
_STATE = "_campaign_late_bound_cut_state"
MAX_CUTS = 12
RPC_TIMEOUT_SECONDS = .025


@dataclass
class CutState:
    request_id: str
    cuts: tuple[int, ...]
    next_cut: int = 0
    ratio_bands: tuple[tuple[float, float], ...] = ()
    used_bands: set[int] = field(default_factory=set)
    reported_inflight_bands: set[int] = field(default_factory=set)
    hold: bool = False
    armed_prefix: int | None = None
    first_held_at: float | None = None
    held_schedule_attempts: int = 0

    @property
    def natural(self):
        return bool(self.ratio_bands)

    @property
    def boundary(self):
        if self.next_cut < len(self.cuts):
            return self.cuts[self.next_cut]
        return None

    def matching_band(self, offload_ratio):
        return next((index for index, (low, high) in enumerate(self.ratio_bands)
                     if index not in self.used_bands and low <= offload_ratio <= high), None)

    def clear_hold(self):
        self.armed_prefix = None
        self.first_held_at = None
        self.held_schedule_attempts = 0


def read_cut_state(request):
    params = request.kv_transfer_params or {}
    contract = params.get(LATE_KEY)
    if contract is None:
        return None
    existing = getattr(request, _STATE, None)
    if existing is not None:
        return existing
    fixed_fields = {"request_id", "cuts"}
    natural_fields = fixed_fields | {"natural", "ratio_bands"}
    if not isinstance(contract, dict) or set(contract) not in (
            fixed_fields, natural_fields, natural_fields | {"hold"}):
        raise ValueError("Invalid late-bound P request contract")
    request_id, cuts = contract["request_id"], contract["cuts"]
    if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
        raise ValueError("Invalid late-bound coordinator request ID")
    length = request.num_prompt_tokens
    bands = ()
    if natural_fields <= set(contract):
        values = contract["ratio_bands"]
        if (contract["natural"] is not True or cuts != [] or not isinstance(values, list)
                or type(contract.get("hold", False)) is not bool
                or not 1 <= len(values) <= MAX_CUTS
                or any(not isinstance(band, list) or len(band) != 2
                       or any(type(bound) not in (float, int) for bound in band)
                       or not 0 < band[0] <= band[1] < 1 for band in values)):
            raise ValueError("Natural cuts require bounded remaining-suffix ratio bands and no clamped cuts")
        bands = tuple(tuple(band) for band in values)
    elif (not isinstance(cuts, list) or not 1 <= len(cuts) <= MAX_CUTS
            or any(type(cut) is not int or not 0 < cut < length for cut in cuts)
            or cuts != sorted(set(cuts))):
        raise ValueError("Late-bound cuts must be strictly increasing internal prompt positions")
    if (params.get("do_remote_decode") is not True or params.get("do_remote_prefill")
            or PLAN_KEY in params or request.max_tokens != 1
            or request.prompt_token_ids is None or len(request.prompt_token_ids) != length
            or not request.has_encoder_inputs or request.resumable):
        raise ValueError("Late-bound path requires a full image prompt and a P-only NIXL producer")
    state = CutState(request_id, tuple(cuts), ratio_bands=bands, hold=contract.get("hold", False))
    setattr(request, _STATE, state)
    return state


def capped_rows(state, computed, proposed):
    if state is None or state.natural or state.boundary is None:
        return proposed
    if computed > state.boundary:
        raise RuntimeError("P scheduler crossed a late-bound cut without making its decision")
    return min(proposed, state.boundary - computed)


def decide_at_cut(request, running_status):
    """Return a ready boundary only after all of this request's GPU writes completed."""
    state = getattr(request, _STATE, None)
    if state is None or state.boundary is None or request.status != running_status:
        return None
    if request.num_in_flight_tokens != 0 or request.num_computed_tokens != state.boundary:
        return None
    if request.num_output_placeholders or request.output_token_ids:
        raise RuntimeError("Late-bound cut reached after a sample was scheduled or generated")
    return state


def natural_offer(request, state, running_status):
    """Find the retired natural prefix even when a later chunk is in flight."""
    if request.status != running_status or request.num_preemptions:
        return None
    prefix = request.num_computed_tokens - request.num_in_flight_tokens
    length = request.num_prompt_tokens
    if not 0 < prefix < length:
        return None
    if state.hold and prefix != state.armed_prefix:
        return None
    band = state.matching_band((length - prefix) / length)
    if band is None:
        return None
    return prefix, band


def arm_natural_hold(request, state, running_status):
    """Arm only after the stock scheduler issues a qualifying chunk."""
    if (not state.hold or state.armed_prefix is not None or request.status != running_status
            or request.num_preemptions or request.num_output_placeholders or request.output_token_ids):
        return
    length, prefix = request.num_prompt_tokens, request.num_computed_tokens
    if 0 < prefix < length and state.matching_band((length - prefix) / length) is not None:
        state.armed_prefix = prefix


def hold_next_rows(request, state, computed):
    if not state.hold or state.armed_prefix is None:
        return False
    if request.num_preemptions or computed != state.armed_prefix:
        state.clear_hold()
        return False
    if state.first_held_at is None:
        state.first_held_at = time.monotonic()
    state.held_schedule_attempts += 1
    return True


class CoordinatorDecision:
    def __init__(self, url, nonce, timeout=RPC_TIMEOUT_SECONDS):
        target = urlsplit(url)
        if (target.scheme != "http" or target.hostname != "127.0.0.1" or target.port is None
                or target.username is not None or target.query or target.fragment
                or target.path != "/campaign/late-decision"):
            raise ValueError("The late-bound decision endpoint must be a local campaign coordinator")
        self.port = target.port
        self.path = target.path
        if not nonce or "\n" in nonce or "\r" in nonce:
            raise ValueError("The late-bound decision endpoint requires its campaign nonce")
        self.nonce = nonce
        self.timeout = timeout

    def __call__(self, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            connection.request("POST", self.path, json.dumps(payload), {
                "Content-Type": "application/json",
                "x-campaign-late-nonce": self.nonce,
            })
            response = connection.getresponse()
            if response.status != 200:
                return False, f"http_{response.status}"
            value = json.loads(response.read(1 << 16))
            if not isinstance(value, dict) or type(value.get("accepted")) is not bool:
                return False, "invalid_response"
            reason = value.get("reason", "accepted" if value["accepted"] else "coordinator_denied")
            return value["accepted"], str(reason)
        except (OSError, ValueError, http.client.HTTPException) as error:
            return False, type(error).__name__
        finally:
            connection.close()


def _free_kv_blocks(scheduler):
    pool = getattr(scheduler.kv_cache_manager, "block_pool", None)
    return int(pool.get_num_free_blocks()) if pool is not None else None


def competitor_work(scheduler, request):
    """Lower-bound remaining prompt rows using scheduler progress including in-flight work."""
    candidates = chain(scheduler.running, scheduler.waiting, scheduler.skipped_waiting)
    work = (
        max(0, other.num_prompt_tokens - other.num_computed_tokens)
        for other in candidates if other is not request
    )
    return [{"remaining_prompt_tokens": count} for count in nsmallest(8, work)]


def install(url):
    """Hook the pinned V1 Scheduler. No vLLM install files are changed."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm.v1.request import RequestStatus

    from ..mllm_epd_prefill_heavy_token_split.common import ROOT, append

    try_schedule_encoder_inputs = Scheduler._try_schedule_encoder_inputs
    if getattr(try_schedule_encoder_inputs, "_campaign_late_bound", False):
        return
    update_after_schedule = Scheduler._update_after_schedule
    update_from_output = Scheduler.update_from_output
    decide = CoordinatorDecision(url, os.environ["MLLM_EPD_RUN_NONCE"])
    log = ROOT / "raw" / f"late_prefill_cut_{os.getpid()}.jsonl"
    skipped_log = ROOT / "raw" / f"late_prefill_natural_skips_{os.getpid()}.jsonl"

    def limit(scheduler, request, num_computed_tokens, num_new_tokens, encoder_compute_budget,
              shift_computed_tokens=0):
        state = read_cut_state(request)
        if state is not None and shift_computed_tokens:
            raise RuntimeError("The late-bound experiment requires no speculation")
        if state is not None and hold_next_rows(request, state, num_computed_tokens):
            return [], 0, encoder_compute_budget, []
        count = capped_rows(state, num_computed_tokens, num_new_tokens)
        if state is not None and count == 0:
            return [], 0, encoder_compute_budget, []
        return try_schedule_encoder_inputs(
            scheduler, request, num_computed_tokens, count, encoder_compute_budget,
            shift_computed_tokens=shift_computed_tokens,
        )

    def after_schedule(scheduler, output):
        update_after_schedule(scheduler, output)
        for request_id in output.num_scheduled_tokens:
            req = scheduler.requests[request_id]
            state = getattr(req, _STATE, None)
            if state is not None and state.hold:
                arm_natural_hold(req, state, RequestStatus.RUNNING)

    def update(scheduler, scheduled, runner_output):
        outputs = update_from_output(scheduler, scheduled, runner_output)
        for request_id, scheduled_tokens in scheduled.num_scheduled_tokens.items():
            req = scheduler.requests.get(request_id)
            if req is None:
                continue
            state = getattr(req, _STATE, None)
            if state is None:
                continue

            band = None
            if state.natural:
                natural = natural_offer(req, state, RequestStatus.RUNNING)
                if natural is None:
                    continue
                prefix, band = natural
                if req.num_in_flight_tokens or req.num_output_placeholders or req.output_token_ids:
                    if band not in state.reported_inflight_bands:
                        state.reported_inflight_bands.add(band)
                        append(skipped_log, {
                            "time": time.monotonic(), "request_id": state.request_id,
                            "prefix_tokens": prefix, "ratio_band": state.ratio_bands[band],
                            "scheduled_chunk_tokens": scheduled_tokens,
                            "later_tokens_inflight": req.num_in_flight_tokens,
                            "reason": "NATURAL_BOUNDARY_PASSED_BY_INFLIGHT_WORK",
                        })
                    continue
            elif decide_at_cut(req, RequestStatus.RUNNING) is not None:
                prefix = state.boundary
            else:
                continue
            observed = time.monotonic()
            payload = {
                "request_id": state.request_id,
                "prefix_tokens": prefix,
                "full_prompt_tokens": req.num_prompt_tokens,
                "prefill_free_kv_blocks": _free_kv_blocks(scheduler),
                "prefill_running": len(scheduler.running),
                "prefill_waiting": len(scheduler.waiting) + len(scheduler.skipped_waiting),
                "competitors": competitor_work(scheduler, req),
                "competitors_observed": True,
                "boundary_ready": observed,
                "lease": {"request_id": state.request_id, "prefix_tokens": prefix},
            }
            if state.natural:
                held_from = state.first_held_at
                payload.update(
                    natural=True, hold=state.hold, ratio_band=state.ratio_bands[band],
                    scheduled_chunk_tokens=scheduled_tokens, held_schedule_attempts=state.held_schedule_attempts,
                    first_held_at=held_from,
                    held_before_retirement_ms=(observed - held_from) * 1000 if held_from is not None else 0.,
                )
            accepted, reason = decide(payload)
            decided = time.monotonic()
            if state.natural:
                state.used_bands.add(band)
                state.clear_hold()
            else:
                state.next_cut += 1
            result = dict(
                payload, accepted=accepted, reason=reason, decision_end=decided,
                decision_ms=(decided - observed) * 1000., engine_request_id=request_id,
                prefills_completed_through=prefix, prefill_recomputations=req.num_preemptions,
                final_prefill_owner="decode" if accepted else "prefill",
            )
            if not accepted:
                # The next scheduling tick uses the same KV block table, request,
                # MRoPE state, and original prompt and multimodal features.
                append(log, result)
                continue

            # Existing _connector_finished() computes the exact prefix block
            # table and registers the original NIXL read and source-block lease.
            scheduler.running.remove(req)
            prefill_stats = req.take_prefill_stats()
            if prefill_stats is not None:
                prefill_stats.finalize(scheduler.kv_cache_manager.estimate_cached_tokens(req))
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            kv, ec = scheduler._free_request(req)
            if (not kv or kv.get("remote_num_tokens") != prefix or not kv.get("remote_block_ids")
                    or not kv.get("do_remote_prefill")):
                raise RuntimeError("Late-bound NIXL export did not cover the actual completed prefix")
            kv[PLAN_KEY] = {"prefix_tokens": prefix, "full_prompt_tokens": req.num_prompt_tokens}
            out = outputs.setdefault(req.client_index, EngineCoreOutputs())
            out.outputs.append(EngineCoreOutput(
                request_id=request_id, new_token_ids=[], finish_reason=req.get_finished_reason(),
                events=req.take_events(), prefill_stats=prefill_stats, kv_transfer_params=kv,
                ec_transfer_params=ec, trace_headers=req.trace_headers,
            ))
            if scheduler.finished_req_ids_dict is not None:
                ids = scheduler.finished_req_ids_dict.pop(req.client_index, set())
                if out.finished_requests is None:
                    out.finished_requests = ids
                else:
                    out.finished_requests.update(ids)
            append(log, dict(result, exported=time.monotonic(), remote_num_tokens=kv["remote_num_tokens"],
                             remote_block_counts=[len(group) for group in kv["remote_block_ids"]]))
        return outputs

    limit._campaign_late_bound = True
    Scheduler._try_schedule_encoder_inputs = limit
    Scheduler._update_after_schedule = after_schedule
    Scheduler.update_from_output = update
