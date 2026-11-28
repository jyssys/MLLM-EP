"""CUPTI observation without Python Scheduler or runner hooks at period zero."""

import os
import time

from ..mllm_epd_prefill_heavy_token_split.cuda_ipc_ec import directory
from ..mllm_epd_prefill_heavy_token_split.worker_observer import CampaignWorkerExtension, Collector


class KernelOnlyCollector(Collector):
    def install_state_hooks(self):
        self.execute_had_instance_slot = "execute_model" in vars(self.runner)
        self.execute_original_slot = vars(self.runner).get("execute_model")

    def close(self):
        # The inherited cleanup deletes a hook when the original is a class method.
        # Supply its temporary slot only while the trace closes.
        if not self.execute_had_instance_slot:
            self.runner.execute_model = self.runner.execute_model
        return super().close()


class ReadyCollector(Collector):
    def install_state_hooks(self):
        super().install_state_hooks()
        if os.environ["MLLM_EPD_ROLE"] != "prefill":
            return
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.request import RequestStatus

        traced_schedule = Scheduler.schedule
        verified = set()

        def readiness(scheduler):
            waiting = list(scheduler.waiting)
            verified.intersection_update(row.request_id for row in waiting)
            ready, missing = 0, 0
            for request in waiting:
                if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                    continue
                if (request.kv_transfer_params or {}).get("do_remote_prefill"):
                    continue
                if request.request_id not in verified:
                    items = request.mm_features or []
                    roots = [directory(scheduler.ec_connector._storage_path, it.identifier) for it in items]
                    if not all((root / "cuda_ipc_ready.json").is_file()
                               and not (root / "cuda_ipc_released.json").exists() for root in roots):
                        missing += 1
                        continue
                    verified.add(request.request_id)
                ready += 1
            return ready, missing

        def schedule(scheduler, *args, **kwargs):
            if time.monotonic_ns() - self.last_schedule_sample_ns < self.state_interval_ns:
                return traced_schedule(scheduler, *args, **kwargs)
            ready, missing = readiness(scheduler)
            measurements = {"kv_allocate_attempts": 0, "kv_allocate_rejected": 0,
                            "ec_availability_attempts": 0, "ec_availability_deferred": 0}
            replaced = []

            def count(method_owner, name, count_key, reject_key, is_rejected):
                original = getattr(method_owner, name)
                original_slot = vars(method_owner).get(name)
                had_slot = name in vars(method_owner)

                def measured(*positional, **keywords):
                    value = original(*positional, **keywords)
                    measurements[count_key] += 1
                    measurements[reject_key] += is_rejected(value)
                    return value

                setattr(method_owner, name, measured)
                replaced.append((method_owner, name, had_slot, original_slot))

            count(scheduler.kv_cache_manager, "allocate_slots",
                  "kv_allocate_attempts", "kv_allocate_rejected", lambda value: value is None)
            if scheduler.ec_connector is not None:
                count(scheduler.ec_connector, "ensure_cache_available",
                      "ec_availability_attempts", "ec_availability_deferred", lambda value: not value)
            try:
                result = traced_schedule(scheduler, *args, **kwargs)
            finally:
                for owner, name, had_slot, slot in reversed(replaced):
                    if had_slot:
                        setattr(owner, name, slot)
                    else:
                        delattr(owner, name)
            if self.rows:
                self.rows[-1]["before"].update(
                    prefill_compute_ready_waiting=ready, prefill_mm_missing_waiting=missing,
                    prefill_waiting_ready_check_scope="Live CUDA IPC E export for every item, ordinary P queue."
                )
                self.rows[-1].update(measurements)
            return result

        Scheduler.schedule = schedule


class DiscoveryWorker(CampaignWorkerExtension):
    def campaign_worker_info(self):
        info = super().campaign_worker_info()
        if os.environ.get("MLLM_EPD_IMAGE_READY") == "1":
            from .image_ready import install_image_ready_observer

            info["native_image_readiness"] = install_image_ready_observer()
        if os.environ.get("MLLM_EPD_NIXL_READY") == "1":
            from .nixl_ready import flush_nixl_ready_observer, install_nixl_ready_observer

            receipt = install_nixl_ready_observer()
            info["native_nixl_readiness"] = flush_nixl_ready_observer() if receipt["enabled"] else receipt
        return info

    def campaign_profile_begin(self, name, include_copies=True, state_interval_ms=0, buffer_gib=1):
        if hasattr(self, "_campaign_collector"):
            raise RuntimeError("One profile per independent worker start")
        collector = ReadyCollector if state_interval_ms > 0 else KernelOnlyCollector
        self._campaign_collector = collector(
            name, os.environ["MLLM_EPD_ROLE"], include_copies, self.model_runner, state_interval_ms, buffer_gib
        )
        return self._campaign_collector.begin()
