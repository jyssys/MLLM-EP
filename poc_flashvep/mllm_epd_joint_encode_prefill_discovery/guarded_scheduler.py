"""Yield between bounded extra D Prefill chunks when native D work is present."""

import os

from ..mllm_epd_modality_aware_idle_execution_discovery.readiness_execution import FRONTIER_KEY
from ..mllm_epd_modality_aware_idle_execution_discovery.readiness_scheduler import ReadinessScheduler


def is_candidate(request):
    return FRONTIER_KEY in (request.kv_transfer_params or {})


class GuardedScheduler(ReadinessScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantum = int(os.environ["MLLM_JOINT_D_QUANTUM_ROWS"])
        if self.role != "decode" or self.quantum not in (128,256,512):
            raise ValueError("Invalid guarded D scheduler geometry")
        self.native_pending = False

    def schedule(self, *args, **kwargs):
        requests = [request for request in self.requests.values() if not request.is_finished()]
        self.native_pending = any(not is_candidate(request) for request in requests)
        if self.native_pending and any(is_candidate(request) for request in requests):
            for queue in (self.waiting, self.skipped_waiting):
                candidates = [request for request in queue if is_candidate(request)]
                for request in candidates:
                    queue.remove_request(request)
                    self.waiting.add_request(request)
        return super().schedule(*args, **kwargs)

    def _try_schedule_encoder_inputs(self, request, num_computed_tokens, num_new_tokens,
                                     encoder_compute_budget, shift_computed_tokens=0):
        if is_candidate(request) and num_computed_tokens < request.num_prompt_tokens:
            contract = self._frontier(request)
            if contract.get("decode_protection") is not True:
                raise ValueError("This D run requires the recipient's decode-protection contract")
            if self.native_pending or request.num_in_flight_tokens:
                return [],0,encoder_compute_budget,[]
            num_new_tokens = min(num_new_tokens,self.quantum)
        return super()._try_schedule_encoder_inputs(request,num_computed_tokens,num_new_tokens,
                                                     encoder_compute_budget,shift_computed_tokens)
