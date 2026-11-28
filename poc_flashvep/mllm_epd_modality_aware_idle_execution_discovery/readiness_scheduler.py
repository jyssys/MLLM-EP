"""Use stock model execution and paged KV on one recipient across a readiness wait."""

import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from ..mllm_epd_prefill_heavy_token_split.common import ROOT, append
from .readiness_execution import FRONTIER_KEY, check_cut, load_contract, published


class ReadinessScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontiers = {}
        self.role = os.environ["MLLM_EPD_ROLE"]
        self.metadata = os.environ["MLLM_EPD_METADATA_PATH"]
        self.encoder_exports = os.environ["MLLM_EPD_EC_PATH"]
        self.trace = ROOT / "raw" / f"ready_retired_{self.role}.jsonl"

    def _frontier(self, request):
        key = (request.kv_transfer_params or {}).get(FRONTIER_KEY)
        if key is None:
            return None
        value = self.frontiers.get(request.request_id)
        if value is None:
            value = load_contract(self.metadata, key)
            positions = [item.mm_position for item in request.mm_features]
            if (value["role"] != self.role or value["prompt_tokens"] != request.num_prompt_tokens
                    or value["images"] != [item.identifier for item in request.mm_features]):
                raise ValueError("Scheduler received a different request from the original frontier")
            check_cut(value, request.prompt_token_ids, [(p.offset, p.offset + p.length) for p in positions])
            self.frontiers[request.request_id] = value
        return value

    def _try_schedule_encoder_inputs(self, request, num_computed_tokens, num_new_tokens,
                                     encoder_compute_budget, shift_computed_tokens=0):
        value = self._frontier(request)
        if value is not None and num_new_tokens and num_computed_tokens < request.num_prompt_tokens:
            if shift_computed_tokens:
                raise ValueError("This exact frontier is scoped to non-speculative serving")
            if num_computed_tokens < value["cut"]:
                num_new_tokens = min(num_new_tokens, value["cut"] - num_computed_tokens)
            elif not value.get("full_published"):
                if all(published(self.encoder_exports, image) for image in value["images"]):
                    value["full_published"] = True
                    value["resume_ns"] = time.monotonic_ns()
                else:
                    return [], 0, encoder_compute_budget, []
        return super()._try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens,
                                                     encoder_compute_budget, shift_computed_tokens)

    def update_from_output(self, scheduler_output, model_runner_output):
        selected = []
        for request_id, n in scheduler_output.num_scheduled_tokens.items():
            if (value := self.frontiers.get(request_id)) is not None:
                request = self.requests.get(request_id)
                if request is None:
                    continue
                before = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
                if before < request.num_prompt_tokens:
                    selected.append((request_id, value, before, n))
        results = super().update_from_output(scheduler_output, model_runner_output)
        now = time.monotonic_ns()
        for request_id, value, before, n in selected:
            request = self.requests.get(request_id)
            finished = request is None or request.is_finished()
            retired = min(value["prompt_tokens"], before + n if finished else
                          max(0, request.num_computed_tokens - request.num_in_flight_tokens))
            append(self.trace, {"request_id": request_id, "key": value["key"], "role": self.role,
                                "retire_ns": now, "before": before, "scheduled_rows": n,
                                "retired_rows": retired, "cut": value["cut"],
                                "resume_ns": value.get("resume_ns"), "finished": finished})
        for request_id in tuple(self.frontiers):
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                del self.frontiers[request_id]
        return results
