"""Issue native Vision forwards for ordered groups and publish only completed CUDA work."""

from dataclasses import replace
import os
from pathlib import Path
from queue import SimpleQueue
import threading
import time

from ..mllm_epd_prefill_heavy_modality_discovery.observer import DiscoveryWorker


SYNC_MODES = ("whole", "1", "2", "4", "8", "16", "v2048", "v4096")
MODES = ("stock",) + SYNC_MODES + tuple(f"a{mode}" for mode in SYNC_MODES)


def partition(items, mode):
    if mode == "whole":
        return [items] if items else []
    token_budget = mode.startswith("v")
    limit = int(mode[1:] if token_budget else mode)
    groups = []
    pending, used = [], 0
    for item in items:
        cost = item[3] if token_budget else 1
        if pending and used + cost > limit:
            groups.append(pending)
            pending, used = [], 0
        pending.append(item)
        used += cost
    if pending:
        groups.append(pending)
    return groups


class GroupedEncoderWorker(DiscoveryWorker):
    def campaign_group_settings(self, mode, signature=False):
        if os.environ["MLLM_EPD_ROLE"] != "encoder" or mode not in MODES or type(signature) is not bool:
            raise ValueError("Invalid experiment group settings")
        from ..mllm_epd_prefill_heavy_token_split.cuda_ipc_ec import REGISTRY

        REGISTRY.capture_correctness_hashes = signature
        if not hasattr(self, "_grouped_encoder"):
            self._grouped_encoder = GroupEncoder(self.model_runner)
        self._grouped_encoder.mode = mode
        return {"mode": mode, "signature": signature,
                "cuda_gate": "A marker is written after the producer observed the CUDA group event complete."}


class GroupEncoder:
    def __init__(self, runner):
        self.runner = runner
        self.original = runner._execute_mm_encoder
        self.mode = "stock"
        self.sequence = 0
        self.publisher = None
        runner._execute_mm_encoder = self.run

    def run(self, scheduler_output):
        if self.mode == "stock" or not scheduler_output.scheduled_encoder_inputs:
            return self.original(scheduler_output)

        import torch
        items = []
        for request_id, indexes in scheduler_output.scheduled_encoder_inputs.items():
            for index in indexes:
                entry = self.runner.requests[request_id].mm_features[index]
                if entry.modality != "image" or entry.data is None:
                    raise ValueError("CUDA grouping is scoped to original image-only input")
                items.append((request_id, index, entry.identifier, entry.mm_position.get_num_embeds()))
        asynchronous = self.mode.startswith("a")
        groups = partition(items, self.mode[1:] if asynchronous else self.mode)
        keys = [item[2] for item in items]
        if len(keys) != len(set(keys)):
            raise ValueError("The measured groups require request-unique original image identities")
        outputs = []
        self.sequence += 1
        if asynchronous and self.publisher is None:
            self.publisher = CompletionPublisher()
        for index, group in enumerate(groups):
            subset = {}
            for request_id, item, _, _ in group:
                subset.setdefault(request_id, []).append(item)
            epoch_ns = time.monotonic_ns()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            part = self.original(replace(scheduler_output, scheduled_encoder_inputs=subset))
            if len(part) != len(group):
                raise ValueError("Native Encoder lost an original image in the group")
            outputs.extend(part)
            issued_ns = time.monotonic_ns()
            end.record()
            descriptor_ids = [item[2] for item in group]
            row = {"sequence": self.sequence, "group": index, "groups": len(groups), "mode": self.mode,
                   "requests": list(subset), "ids": descriptor_ids, "visual_rows": sum(item[3] for item in group),
                   "submit_ns": epoch_ns, "native_saved_ns": issued_ns}
            if asynchronous:
                self.publisher.submit(start, end, row)
                continue
            if end.query():
                incomplete_ns = epoch_ns
            else:
                incomplete_ns = issued_ns
                end.synchronize()
            finish(start, end, row, incomplete_ns)
        return outputs


def finish(start, end, row, lower_ns):
    from ..mllm_epd_prefill_heavy_token_split.common import ROOT, append, write
    from ..mllm_epd_prefill_heavy_token_split.cuda_ipc_ec import directory

    confirmed_ns = time.monotonic_ns()
    ec = Path(os.environ["MLLM_EPD_EC_PATH"])
    row.update(cuda_ready_lower_ns=lower_ns, cuda_ready_upper_ns=confirmed_ns,
               gpu_elapsed_ms=start.elapsed_time(end))
    for identifier in row["ids"]:
        folder = directory(ec, identifier)
        if not (folder / "cuda_ipc_inline_prefill.json").is_file():
            raise ValueError("Original IPC descriptor is absent after native group publication")
        write(folder / "cuda_group_ready.json", {"identifier": identifier, "ready_ns": confirmed_ns})
    row["published_ns"] = time.monotonic_ns()
    append(ROOT / "raw" / "group_encoder.jsonl", row)


class CompletionPublisher:
    """Publish ordered positive CUDA completions without synchronizing the E engine."""

    def __init__(self):
        self.queue = SimpleQueue()
        self.error = None
        threading.Thread(target=self.run, name="epd-group-cuda-publisher", daemon=True).start()

    def submit(self, start, end, row):
        if self.error is not None:
            raise RuntimeError("CUDA-ready publisher stopped") from self.error
        self.queue.put((start,end,row))

    def run(self):
        import torch

        try:
            torch.cuda.set_device(0)
            while True:
                start,end,row = self.queue.get()
                lower_ns = row["native_saved_ns"]
                while True:
                    now = time.monotonic_ns()
                    if end.query():
                        break
                    lower_ns = now
                    time.sleep(.001)
                finish(start,end,row,lower_ns)
        except Exception as exc:
            self.error = exc
