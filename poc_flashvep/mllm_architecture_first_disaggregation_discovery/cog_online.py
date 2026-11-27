"""Finite open-loop CogVLM2 pilot with real P→D KV import and dynamic decode batches."""
import argparse
from collections import Counter
from dataclasses import dataclass, field
import importlib.metadata
import json
from pathlib import Path
from queue import Empty, Queue
import random
import threading
import time

import numpy as np
import torch
from torch.nn.functional import pad
from transformers import AutoTokenizer

from .hard_expert_decode import (
    build_model, checked_snapshot, decode_step, exact_copy, generic_inactive_modules,
    import_snapshot, prefill, sha,
)

SEED = 2026112707
CAP = 2048
MODES = ("dp2", "resident_text_fast", "visual_pruned", "generic_pruned")
SLO = {"ttft_ms": 5000, "itl_ms": 34, "tpot_ms": 33}


@dataclass
class Request:
    original: dict
    arrival: float
    dispatch: float = 0.
    prefill_start: float = 0.
    prefill_end: float = 0.
    ready: float = 0.
    kv_import_ms: float = 0.
    input_tokens: int = 0
    last_position: int = 0
    tokens: list = field(default_factory=list)
    emissions: list = field(default_factory=list)
    finish: str = ""

    @property
    def current_length(self):
        return self.input_tokens + len(self.tokens) - 1

    def emit(self, token, timestamp, eos, cap):
        self.tokens.append(int(token))
        self.emissions.append(timestamp)
        if token in eos:
            self.finish = "eos"
        elif len(self.tokens) == cap:
            self.finish = "cap"

    def record(self, tokenizer):
        deltas = [(right - left) * 1000 for left, right in zip(self.emissions, self.emissions[1:])]
        ttft = (self.emissions[0] - self.arrival) * 1000
        tpot = sum(deltas) / len(deltas) if deltas else 0.
        return {
            "request_id": self.original["request_id"], "prompt_sha256": self.original["prompt_SHA256"],
            "source": self.original["source"], "tokens": self.tokens, "input_tokens": self.input_tokens,
            "last_prompt_position": self.last_position, "ttft_ms": ttft, "itl_ms": deltas, "tpot_ms": tpot,
            "e2e_ms": (self.emissions[-1] - self.arrival) * 1000, "finish": self.finish,
            "joint_slo": ttft <= SLO["ttft_ms"] and max(deltas, default=0) <= SLO["itl_ms"]
                         and tpot <= SLO["tpot_ms"],
            "arrival": self.arrival, "dispatch": self.dispatch, "prefill_start": self.prefill_start,
            "prefill_end": self.prefill_end, "ready": self.ready, "emissions": self.emissions,
            "kv_import_ms": self.kv_import_ms,
            "text": tokenizer.decode(self.tokens, skip_special_tokens=True),
        }


def add_cache(current, incoming):
    caches = ([current] if current is not None else []) + incoming
    if len(caches) == 1:
        return caches[0]
    max_len = max(cache[0][0].shape[2] for cache in caches)
    packed = tuple(
        tuple(torch.cat([
            pad(t, (0, 0, max_len - t.shape[2], 0)) if t.shape[2] < max_len else t
            for t in values
        ], dim=0) for values in zip(*layer, strict=True))
        for layer in zip(*caches, strict=True)
    )
    return packed


def select_cache(cache, rows, device):
    chosen = torch.tensor(rows, device=device, dtype=torch.long)
    return tuple(tuple(value.index_select(0, chosen) for value in layer) for layer in cache)


class DecodeBatch:
    def __init__(self, model, definition, cap, device):
        self.model, self.definition, self.device, self.cap = model, definition, device, cap
        eos = model.config.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos)
        self.active, self.cache = [], None
        self.batch_histogram = Counter()
        self.slot_depth = []

    def admit(self, prepared, timestamp):
        additions = []
        for request, packet in prepared:
            begin = time.monotonic()
            kv = packet["kv"]
            if kv[0][0].device != self.device:
                kv = exact_copy(kv, self.device)
                request.kv_import_ms = (time.monotonic() - begin) * 1000
            request.input_tokens = packet["model_input_tokens"]
            request.last_position = packet["last_position"]
            request.ready = time.monotonic()
            request.emit(packet["first_id"][0].item(), request.ready, self.eos, self.cap)
            if not request.finish:
                self.active.append(request)
                additions.append(kv)
        if additions:
            self.cache = add_cache(self.cache, additions)
        self.slot_depth.append({"time": timestamp, "active": len(self.active), "admitted": len(prepared)})

    def step(self):
        if not self.active:
            return
        count = len(self.active)
        past = self.cache[0][0].shape[2]
        lengths = torch.tensor([request.current_length for request in self.active], device=self.device)
        tokens = torch.tensor([[request.tokens[-1]] for request in self.active], device=self.device)
        positions = torch.tensor([[request.last_position + len(request.tokens)] for request in self.active],
                                 device=self.device)
        mask = torch.arange(past + 1, device=self.device)[None] >= (past - lengths[:, None])
        logits, cache = decode_step(self.model, self.definition, tokens, self.cache, positions, mask)
        ids = logits.argmax(-1).tolist()
        timestamp = time.monotonic()
        self.batch_histogram[count] += 1
        for request, token in zip(self.active, ids, strict=True):
            request.emit(token, timestamp, self.eos, self.cap)
        remaining = [index for index, request in enumerate(self.active) if not request.finish]
        if len(remaining) == count:
            self.cache = cache
        elif remaining:
            self.cache = select_cache(cache, remaining, self.device)
        else:
            self.cache = None
        self.active = [self.active[index] for index in remaining]


def prepared(model, definition, tokenizer, request):
    request.prefill_start = time.monotonic()
    packet = prefill(model, tokenizer, definition, request.original)
    request.prefill_end = time.monotonic()
    return request, packet


def generate(device, errors, worker, *args):
    def run():
        try:
            with torch.inference_mode(), torch.cuda.device(device):
                worker(*args)
        except BaseException as error:
            errors.put(error)
    return threading.Thread(target=run, daemon=True)


def p_worker(model, definition, tokenizer, incoming, ready):
    while (request := incoming.get()) is not None:
        ready.put(prepared(model, definition, tokenizer, request))
    ready.put(None)


def execute_batch(batch, incoming, slots, prepare=None):
    """Admit up to the same fixed slot cap and advance each active request one native step."""
    closed = False
    while not closed or batch.active:
        arrivals = []
        # For co-location, emit the first local result as soon as its prefill
        # completes and give any active rows an iteration before more local P.
        limit = 1 if prepare is not None else slots - len(batch.active)
        while not closed and len(batch.active) < slots and len(arrivals) < limit:
            try:
                item = incoming.get() if not batch.active and not arrivals else incoming.get_nowait()
            except Empty:
                break
            if item is None:
                closed = True
                break
            arrivals.append(prepare(item) if prepare is not None else item)
        if arrivals:
            batch.admit(arrivals, time.monotonic())
        batch.step()


def timed_arrivals(rows, qps):
    rng = random.Random(SEED)
    elapsed = 0.
    plan = []
    for index, row in enumerate(rows):
        if index:
            elapsed += rng.expovariate(qps)
        plan.append((row, elapsed))
    return plan


def cohort(mode, models, definition, tokenizer, rows, qps, slots, cap):
    errors = Queue()
    plan = timed_arrivals(rows, qps)
    start = time.monotonic() + .5
    requests = [Request(row, start + offset) for row, offset in plan]
    if mode == "dp2":
        queues = [Queue(), Queue()]
        assignments = [[], []]
        batches = [DecodeBatch(model, definition, cap, torch.device(f"cuda:{index}"))
                   for index, model in enumerate(models)]

        def adapter(model):
            return lambda request: prepared(model, definition, tokenizer, request)

        threads = [generate(index, errors, execute_batch, batch, inbox, slots, adapter(model))
                   for index, (batch, inbox, model) in enumerate(zip(batches, queues, models, strict=True))]
    else:
        queues = [Queue()]
        ready = Queue()
        batches = [DecodeBatch(models[1], definition, cap, torch.device("cuda:1"))]
        threads = [
            generate(0, errors, p_worker, models[0], definition, tokenizer, queues[0], ready),
            generate(1, errors, execute_batch, batches[0], ready, slots),
        ]
    for thread in threads:
        thread.start()
    for idx, request in enumerate(requests):
        while True:
            if not errors.empty():
                raise errors.get()
            remaining = request.arrival - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, .1))
        request.dispatch = time.monotonic()
        rank = 0
        if mode == "dp2":
            # Outstanding work includes requests currently in prefill.
            rank = min(range(2), key=lambda i: (sum(not req.finish for req in assignments[i]), (i - idx) % 2))
            assignments[rank].append(request)
        queues[rank].put(request)
    for queue in queues:
        queue.put(None)
    while any(thread.is_alive() for thread in threads):
        if not errors.empty():
            raise errors.get()
        for thread in threads:
            thread.join(.1)
    if not errors.empty():
        raise errors.get()
    if not all(request.finish for request in requests):
        raise RuntimeError("Incomplete pilot cohort")
    results = [request.record(tokenizer) for request in requests]
    duration = max(item["emissions"][-1] for item in results) - start
    histogram = sum((batch.batch_histogram for batch in batches), Counter())
    itls = [gap for item in results for gap in item["itl_ms"]]
    good = [item for item in results if item["joint_slo"]]

    def q(field, percentile):
        return float(np.percentile([item[field] for item in results], percentile))

    return {
        "mode": mode, "offered_qps": qps, "cap": cap, "configured_slots_per_decode_worker": slots,
        "planned_arrivals": [{"request_id": row["request_id"], "offset": offset} for row, offset in plan],
        "duration_seconds": duration, "completed": len(results), "requests_per_s": len(results) / duration,
        "output_tokens_per_s": sum(len(item["tokens"]) for item in results) / duration,
        "slo_goodput_requests_per_s": len(good) / duration,
        "slo_goodput_tokens_per_s": sum(len(item["tokens"]) for item in good) / duration,
        "joint_slo": len(good) / len(results), "SLO": SLO,
        "ttft_p50_ms": q("ttft_ms", 50), "ttft_p95_ms": q("ttft_ms", 95),
        "tpot_p95_ms": q("tpot_ms", 95), "itl_p99_ms": float(np.percentile(itls, 99)) if itls else 0.,
        "e2e_p95_ms": q("e2e_ms", 95), "batch_histogram": dict(histogram),
        "step_weighted_decode_batch": sum(batch * n for batch, n in histogram.items()) / histogram.total()
                                      if histogram else 0.,
        "slot_events": [batch.slot_depth for batch in batches], "requests": results,
        "host_boundary": "Actual open-loop arrivals with emitted token IDs, in process. "
                         "P includes preprocessing and first-logit execution. No HTTP, paged scheduler or production allocator.",
    }


def write(path, item):
    with path.open("x") as stream:
        json.dump(item, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--qps", type=float, required=True, nargs="+")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    if args.slots < 1 or any(qps <= 0 for qps in args.qps):
        parser.error("Specify positive load and active slots")
    config, checkpoint = checked_snapshot(args.snapshot)
    definition, attention = import_snapshot(args.snapshot)
    inactive = generic_inactive_modules(args.snapshot / "modeling_cogvlm.py")
    layouts = ("monolith_text_fast",) * 2 if args.mode == "dp2" else ("P", args.mode)
    models, fingerprints = [], []
    for index, layout in enumerate(layouts):
        with torch.cuda.device(index):
            model, footprint = build_model(definition, config, args.snapshot, checkpoint,
                                          layout, f"cuda:{index}", inactive)
        models.append(model)
        fingerprints.append({"layout": layout, "weight_bytes": sum(row["bytes"] for row in footprint.values()),
                             "allocator_current_bytes": torch.cuda.memory_allocated(index)})
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    rows = json.loads(args.manifest.read_text())["requests"][:args.count]
    if not rows or any(row["image_count"] != 1 or row["study_split"] != "discovery" for row in rows):
        raise ValueError("Pilot requires an intact one-image Discovery manifest")
    random.Random(SEED + 11).shuffle(rows)
    args.out.mkdir(parents=True)
    write(args.out / "environment.json", {
        "mode": args.mode, "revisions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "checkpoint_config_sha256": sha(args.snapshot / "config.json"),
        "runner_sha256": sha(Path(__file__)),
        "manifest_sha256": sha(args.manifest), "compiled_attention": attention,
        "model_footprints": fingerprints, "seed": SEED, "cap": CAP,
    })
    # Warm on the same frozen cohort, outside the measured Natural run.
    cohort(args.mode, models, definition, tokenizer, rows[: min(2, len(rows))], 6., args.slots, 3)
    for index, qps in enumerate(args.qps):
        for device in (0, 1):
            torch.cuda.reset_peak_memory_stats(device)
        result = cohort(args.mode, models, definition, tokenizer, rows, qps, args.slots, CAP)
        result["peak_allocated_bytes_by_device"] = [torch.cuda.max_memory_allocated(i) for i in (0, 1)]
        write(args.out / f"cohort_{index + 1}.json", result)
        print(args.mode, qps, result["completed"], result["joint_slo"], result["ttft_p95_ms"], flush=True)


if __name__ == "__main__":
    main()
