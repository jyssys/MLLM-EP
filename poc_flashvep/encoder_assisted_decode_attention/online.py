"""Open-loop three-H200 EADA with real Qwen model stages and paged FA3."""
import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from queue import Empty, Queue
import random
import threading
import time

import numpy as np
import torch

from .cache import Arena, DecodePages, Placement, partition
from .model import Checkpoint, TokenDecoder, stage_weights
from .peer_attention import AssistedAttention
from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt


SLO = {"ttft_ms": 5000, "all_itl_ms": 34, "tpot_ms": 33}


@dataclass
class Request:
    ordinal: int
    row: dict
    planned: float
    metadata: dict = field(default_factory=dict)
    tokens: list = field(default_factory=list)
    emissions: list = field(default_factory=list)
    placement: object = None
    finish: str = ""
    position_delta: int = 0
    input_tokens: int = 0

    def emit(self, value, timestamp, eos, cap, *, natural):
        self.tokens.append(int(value))
        self.emissions.append(timestamp)
        if natural and value in eos:
            self.finish = "eos"
        elif len(self.tokens) >= cap:
            self.finish = "cap"


@dataclass
class Packet:
    request: Request
    cache: object
    first: int
    visual: torch.Tensor
    rank: int


def trace(rows, count, qps, seed):
    if count <= 0 or qps <= 0:
        raise ValueError("Trace requires positive requests and offered rate")
    rng = random.Random(seed)
    clock = 0
    n = 0
    while n < count:
        content = rows.copy()
        rng.shuffle(content)
        for row in content:
            if n >= count:
                break
            if n:
                clock += rng.expovariate(qps)
            yield row, clock
            n += 1


def checked_rows(args):
    rows = rt.read(args.manifest)["requests"]
    source = "LongBench" if args.workload == "real-text" else "Main100"
    rows = [r for r in rows if r.get("study_source", "Main100") == source]
    allowed = {"single": (1, 1), "few": (2, 4), "many": (5, 1000), "mixed": (1, 1000)}
    lo, hi = allowed[args.images]
    if source == "LongBench":
        lo = hi = 0
    rows = [r for r in rows if lo <= r["image_count"] <= hi and r["total_input_tokens"] <= args.input_cap]
    if not rows or len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError(f"A nonempty, unique frozen {source} roster is required")
    if args.workload == "realistic":
        original = {}
        for row in rows:
            for datum in row["images"] + [{"path": row["token_metadata_file"], "SHA256": row["token_metadata_SHA256"]}]:
                original.setdefault(datum["path"], datum["SHA256"])
        for path, contents in original.items():
            if rt.sha(Path(path)) != contents:
                raise ValueError(f"Original source metadata mismatch for {path}")
    return rows


def launch(target, args, device, errors):
    def guard():
        try:
            with torch.inference_mode(), torch.cuda.device(device):
                target(*args)
        except BaseException as error:
            errors.put(error)
    return threading.Thread(target=guard, daemon=True)


class Session:
    def __init__(self, args):
        self.args = args
        self.checkpoint = Checkpoint(args.model, args.family)
        self.pranks = {"epd": [1], "dual-prefill": [0, 1], "dual-decode": [0]}[args.topology]
        self.dranks = [1, 2] if args.topology == "dual-decode" else [2]
        e_device = "cuda:0" if args.topology == "epd" and args.workload == "realistic" else None
        self.prefill_models = {
            rank: self.checkpoint.replica(torch.device("cuda", rank), vision_device=(
                e_device or (torch.device("cuda", rank) if args.workload == "realistic" else None)),
                                         packed_vision=args.vision_backend == "packed-fa3")
            for rank in self.pranks
        }
        self.encoder = self.prefill_models[1].model.visual if e_device is not None else None
        self.decoders = {
            rank: TokenDecoder(self.checkpoint, self.checkpoint.replica(torch.device("cuda", rank)),
                               torch.device("cuda", rank)) for rank in self.dranks
        }
        self.weights = {rank: stage_weights(model) for rank, model in self.prefill_models.items()}
        if self.encoder is not None:
            self.weights[0] = stage_weights(self.encoder)
            self.weights[1] -= self.weights[0]
        for rank, stage in self.decoders.items():
            self.weights[rank] = stage_weights(stage.text) + stage_weights(stage.lm_head)
        self.encoder_active = threading.Event()
        self.renderer_active = threading.Event()
        self.helper_rank = {"none": None, "encoder": 0, "prefill": self.pranks[0],
                            "idle-encoder": 0}[args.helper]
        first = next(iter(self.decoders.values()))
        geometry = (first.n_layers, first.kv_heads, first.dim)
        self.arenas = {rank: Arena(torch.device("cuda", rank), *geometry, args.d_gib)
                       for rank in self.dranks}
        self.helper = (Arena(torch.device("cuda", self.helper_rank), *geometry, args.helper_gib)
                       if self.helper_rank is not None else None)
        self.ops = {}
        self.graphs = {}
        if args.attention_engine == "graph":
            from .paged_graph import PagedGraphPool
            helper_limit = args.graph_context if args.partition in ("visual_all", "matched_visual_count") else max(
                128, 1 << (max(1, int(args.graph_context * args.ratio)) - 1).bit_length())
            self.graphs = {rank: PagedGraphPool(self.arenas[rank], self.helper, heads=stage.heads, dim=stage.dim,
                                               max_batch=args.max_batch, max_bundles=args.graph_bundles,
                                               max_context=65536, fixed_d_context=args.graph_context,
                                               fixed_helper_context=helper_limit)
                           for rank, stage in self.decoders.items()}
            for rank, pool in self.graphs.items():
                with torch.cuda.device(rank):
                    pool.prewarm(range(1, args.max_batch + 1), include_local=True, include_split=self.helper is not None)

    def render_worker(self, source, ready):
        while (request := source.get()) is not None:
            self.renderer_active.set()
            request.metadata["e_start"] = time.monotonic()
            item = self.checkpoint.preprocess(request.row, workload=self.args.workload)
            request.metadata["render_end"] = time.monotonic()
            ready.put((request,item))
            self.renderer_active.clear()
        ready.put(None)

    def encode_worker(self, source, ready):
        stream = torch.cuda.Stream(device=0, priority=-1)
        while (bundle := source.get()) is not None:
            if self.args.encoder_pipeline:
                request,item = bundle
            else:
                request = bundle
                request.metadata["e_start"] = time.monotonic()
                item = self.checkpoint.preprocess(request.row, workload=self.args.workload)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                self.encoder_active.set()
                request.metadata["e_gpu_begin"] = time.monotonic()
                try:
                    start.record()
                    images = self.checkpoint.encode(self.encoder, item, "cuda:0")
                    end.record()
                    end.synchronize()
                finally:
                    self.encoder_active.clear()
            request.metadata["e_end"] = time.monotonic()
            request.metadata["e_gpu_enclosure_ms"] = start.elapsed_time(end)
            ready.put((request, item, images))
        ready.put(None)

    def prefill_worker(self, rank, source, ready):
        dev = torch.device("cuda", rank)
        model = self.prefill_models[rank]
        while (bundle := source.get()) is not None:
            if self.encoder is not None:
                request, item, images = bundle
            else:
                request, images = bundle, None
                if self.args.workload == "realistic":
                    request.metadata["e_start"] = time.monotonic()
                item = self.checkpoint.preprocess(request.row, workload=self.args.workload)
                if self.args.workload == "realistic":
                    request.metadata["e_gpu_begin"] = time.monotonic()
                    images = self.checkpoint.encode(model.model.visual, item, dev)
                    torch.cuda.current_stream(dev).synchronize()
                    request.metadata["e_end"] = time.monotonic()
            request.metadata["p_start"] = time.monotonic()
            request.metadata["source_p"] = rank
            prefix, first = self.checkpoint.prefill(model, item, images, dev, verify_rope=self.warming)
            request.metadata["p_end"] = time.monotonic()
            request.input_tokens = item["ids"].shape[-1]
            request.position_delta = item["delta"]
            ready.put(Packet(request, prefix, first, item["visual"], rank))
        ready.put(None)

    def allocate(self, decoder, packet, cap):
        request = packet.request
        begin = time.monotonic()
        local = self.arenas[decoder]
        mode = self.args.partition if self.helper is not None else "local"
        requested_mode = mode
        if self.args.helper == "idle-encoder" and (self.encoder_active.is_set() or self.e_inbox.qsize()
                or self.renderer_active.is_set() or self.e_prepared.qsize()):
            mode = "local"
        if self.args.helper == "prefill" and packet.rank != self.helper_rank:
            # L2 retains source-side pages on the assisting P. The other P remains a native route.
            mode = "local"
        d_idx, helper_idx = partition(request.input_tokens, packet.visual, mode, self.args.ratio)
        if not local.fits(d_idx.numel() + cap) or (helper_idx.numel() and not self.helper.fits(helper_idx.numel())):
            request.metadata["page_waits"] = request.metadata.get("page_waits", 0) + 1
            return False
        d_pages = local.acquire(d_idx.numel() + cap)
        helper_pages = self.helper.acquire(helper_idx.numel()) if helper_idx.numel() else []
        move_d = local.enqueue(packet.cache, d_idx, d_pages)
        move_h = self.helper.enqueue(packet.cache, helper_idx, helper_pages) if helper_pages else None
        duration_d, duration_h = local.wait(move_d), local.wait(move_h)
        request.placement = Placement(d_pages, helper_pages, d_idx.numel(), helper_idx.numel(), packet.rank, mode)
        done = time.monotonic()
        per_row = local.bytes_per_row
        request.metadata.update({"d_gpu": decoder, "d_admission": begin, "d_ready": done,
                                 "initial_host_ms": (done - begin) * 1000,
                                 "initial_d_enclosure_ms": duration_d,
                                 "initial_helper_enclosure_ms": duration_h, "policy_requested": requested_mode,
                                 "policy_actual": mode, "local_prompt_rows": d_idx.numel(),
                                 "helper_prompt_rows": helper_idx.numel(),
                                 "p_to_d_logical_bytes": d_idx.numel()*per_row,
                                 "p_to_helper_logical_bytes": (helper_idx.numel()*per_row
                                                                if packet.rank != self.helper_rank else 0),
                                 "helper_retained_bytes": helper_idx.numel()*per_row})
        request.emit(packet.first, done, self.checkpoint.eos, cap, natural=not self.args.output)
        return True

    def release(self, rank, request):
        self.arenas[rank].release(request.placement.d_pages)
        if request.placement.helper_pages:
            self.helper.release(request.placement.helper_pages)

    def decode_worker(self, rank, source, cap, stats, closed_after):
        decoder = self.decoders[rank]
        active, waiting, closed = [], None, 0
        max_batch = self.args.max_batch
        while closed < closed_after or active or waiting:
            # One ingest before the next active decode step bounds host admission bursts.
            while len(active) < max_batch and (closed < closed_after or waiting):
                try:
                    packet = waiting or (source.get() if not active else source.get_nowait())
                except Empty:
                    break
                if packet is None:
                    closed += 1
                    continue
                if not self.allocate(rank, packet, cap):
                    if not active:
                        packet.request.finish = "admission_failure"
                        packet.request.metadata["d_rejected"] = time.monotonic()
                        waiting = None
                        continue
                    waiting = packet
                    break
                waiting = None
                if packet.request.finish:
                    self.release(rank, packet.request)
                else:
                    active.append(packet.request)
                    break
            if not active:
                continue
            batch = len(active)
            key = (rank, batch)
            assists = any(r.placement.helper_rows for r in active)
            if assists and key not in self.ops:
                self.ops[key] = AssistedAttention((batch, decoder.heads, decoder.dim),
                                                  decode_device=rank, helper_device=self.helper_rank)
            pages = DecodePages(self.arenas[rank], self.helper, active, self.ops.get(key),
                                decoder.heads, decoder.dim)
            begin = time.monotonic()
            if rank in self.graphs:
                pages.graph_batch = self.graphs[rank].prepare(pages)
            state = self.encoder_active.is_set()
            next_tokens, _, measurements = decoder(
                [r.tokens[-1] for r in active],
                [r.input_tokens + r.position_delta + len(r.tokens) - 1 for r in active],
                pages, instrument=self.args.instrument)
            end = time.monotonic()
            row = {"rank": rank, "begin": begin, "end": end, "batch": batch,
                   "ordinals": [r.ordinal for r in active], "assisted": assists,
                   "attention_backend": "graph" if pages.graph_batch is not None else "eager",
                   "encoder_active_at_launch": state, "shards": [pages.d_lengths, pages.h_lengths]}
            if assists:
                row["D_helper_logical_bytes"] = decoder.n_layers * sum(self.ops[key].link_bytes.values())
            if measurements:
                row.update(measurements)
            stats.append(row)
            for r, new in zip(active, next_tokens, strict=True):
                r.emit(new, end, self.checkpoint.eos, cap, natural=not self.args.output)
                if r.finish:
                    self.release(rank, r)
            active = [r for r in active if not r.finish]

    def cohort(self, rows, count, qps, seed, cap, *, warming=False, arrival_plan=None):
        self.warming = warming
        self.e_inbox = Queue()
        self.e_prepared = Queue(maxsize=2)
        e_done = Queue(maxsize=4)
        p_inboxes = [e_done] if self.encoder is not None else [Queue() for _ in self.pranks]
        d_queue = Queue(maxsize=4)
        d_inboxes = [d_queue for _ in self.dranks]
        errors = Queue()
        steps = []
        for arena in (*self.arenas.values(), *([self.helper] if self.helper is not None else [])):
            if arena.owners:
                raise ValueError("Prior cohort still owns KV pages")
            arena.peak_used = 0
        for index in range(3):
            torch.cuda.reset_peak_memory_stats(index)
        threads = [launch(self.prefill_worker, (rank, inbox, d_queue), rank, errors)
                   for rank, inbox in zip(self.pranks, p_inboxes, strict=True)]
        if self.encoder is not None:
            ready = self.e_inbox
            if self.args.encoder_pipeline:
                threads.append(launch(self.render_worker,(self.e_inbox,self.e_prepared),0,errors))
                ready = self.e_prepared
            threads.append(launch(self.encode_worker, (ready, e_done), 0, errors))
        shared_end = self.args.topology == "dual-decode"
        if shared_end:
            # With one producer and multiple consumers, fan out one end marker per consumer.
            def fanout():
                while (packet := d_queue.get()) is not None:
                    worker = min(range(len(local_inboxes)),
                                 key=lambda i: (sum(not r.finish for r in assigned[i]), i))
                    assigned[worker].append(packet.request)
                    local_inboxes[worker].put(packet)
                for inbox in local_inboxes:
                    inbox.put(None)
            local_inboxes = [Queue(maxsize=4) for _ in self.dranks]
            assigned = [[] for _ in self.dranks]
            threads.append(launch(fanout, (), self.pranks[0], errors))
            d_inboxes = local_inboxes
        threads += [launch(self.decode_worker, (rank, inbox, cap, steps,
                                                  1 if shared_end else len(self.pranks)), rank, errors)
                    for rank, inbox in zip(self.dranks, d_inboxes, strict=True)]
        for thread in threads:
            thread.start()
        start = time.monotonic() + .25
        plan = arrival_plan if arrival_plan is not None else trace(rows, count, qps, seed)
        scheduled = [Request(i, row, start + offset) for i, (row, offset) in enumerate(plan)]
        assigned_p = [[] for _ in self.pranks]
        for request in scheduled:
            while time.monotonic() < request.planned:
                if not errors.empty():
                    raise errors.get()
                time.sleep(min(.04, max(0., request.planned - time.monotonic())))
            request.metadata["arrived"] = time.monotonic()
            if self.encoder is not None:
                self.e_inbox.put(request)
            else:
                slot = min(range(len(p_inboxes)),
                           key=lambda i: (sum("p_end" not in r.metadata for r in assigned_p[i]),
                                          (i - request.ordinal) % len(p_inboxes)))
                assigned_p[slot].append(request)
                p_inboxes[slot].put(request)
        for box in ([self.e_inbox] if self.encoder is not None else p_inboxes):
            box.put(None)
        while any(thread.is_alive() for thread in threads):
            if not errors.empty():
                raise errors.get()
            for thread in threads:
                thread.join(.05)
        if not errors.empty():
            raise errors.get()
        if not all(r.finish for r in scheduled):
            raise RuntimeError("Closed cohort lost a request")
        return summarize(scheduled, steps, start, qps, seed, self, cap)


def summarize(scheduled, steps, start, qps, seed, session, cap):
    raw = []
    for request in scheduled:
        times = [(right-left)*1e3 for left, right in zip(request.emissions, request.emissions[1:])]
        item = {"id": request.row["request_id"], "ordinal": request.ordinal,
                "source": request.row["source"], "planned": request.planned,
                "image_count": request.row["image_count"], "input_rows": request.input_tokens,
                "original_visual_rows": request.row["visual_tokens"], "output_ids": request.tokens,
                "emissions": request.emissions, "outcome": request.finish, **request.metadata}
        if request.emissions:
            ttft = (request.emissions[0] - request.planned)*1e3
            tpot = sum(times)/len(times) if times else 0.
            item.update({"ttft_ms": ttft, "itl_ms": times, "tpot_ms": tpot,
                         "e2e_ms": (request.emissions[-1]-request.planned)*1e3})
            item["joint_slo"] = (ttft <= SLO["ttft_ms"] and tpot <= SLO["tpot_ms"]
                                 and max(times, default=0.) <= SLO["all_itl_ms"])
        else:
            item["joint_slo"] = False
        raw.append(item)
    done = [r for r in raw if r["outcome"] in ("eos", "cap")]
    elapsed = max([r.get("d_rejected", start) for r in raw] + [r["emissions"][-1] for r in done]) - start
    pools = [a.receipt() for a in session.arenas.values()]
    if session.helper is not None:
        pools.append(session.helper.receipt())
    hist = Counter(item["batch"] for item in steps)
    vals = {"ttft_ms":[r["ttft_ms"] for r in done], "tpot_ms":[r["tpot_ms"] for r in done],
            "itl_ms":[gap for r in done for gap in r["itl_ms"]], "e2e_ms":[r["e2e_ms"] for r in done]}
    percentiles = {name: {f"p{q}": float(np.percentile(items,q)) if items else None for q in (50,95,99)}
                   for name,items in vals.items()}
    return {"evidence": "MEASURED_ONLINE", "boundary": "Open loop in process, explicit three-GPU real model stages",
            "topology": session.args.topology, "family": session.args.family, "helper": session.args.helper,
            "partition": session.args.partition, "requested_ratio": session.args.ratio, "workload":session.args.workload,
            "attention_engine": session.args.attention_engine,
            "vision_backend": session.args.vision_backend, "encoder_pipeline": session.args.encoder_pipeline,
            "graph_stats": {rank: pool.receipt() for rank, pool in session.graphs.items()},
            "gpu_profiler": session.args.instrument, "offered_qps": qps, "seed": seed, "cap": cap,
            "start_monotonic": start, "makespan_seconds": elapsed, "outcomes": dict(Counter(r["outcome"] for r in raw)),
            "complete": len(done), "planned": len(raw), "SLO": SLO, "latencies": percentiles,
            "requests_per_second":len(done)/elapsed, "tokens_per_second":sum(len(r["output_ids"]) for r in done)/elapsed,
            "goodput_requests_per_second":sum(r["joint_slo"] for r in raw)/elapsed,
            "joint_slo":sum(r["joint_slo"] for r in raw)/len(raw), "batch_histogram": dict(hist),
            "allocated_weight_bytes": session.weights, "stage_kv_pools": pools,
            "peak_process_allocated_bytes": [torch.cuda.max_memory_allocated(d) for d in range(3)],
            "peak_process_reserved_bytes": [torch.cuda.max_memory_reserved(d) for d in range(3)],
            "requests": raw, "decode_steps": steps}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--family", choices=["qwen-vl", "qwen-text"], default="qwen-vl")
    p.add_argument("--workload", choices=["realistic","controlled-text","real-text"], default="realistic")
    p.add_argument("--images", choices=["mixed","single","few","many"], default="mixed")
    p.add_argument("--input-cap", type=int, default=32768)
    p.add_argument("--topology", choices=["epd","dual-prefill","dual-decode"], default="epd")
    p.add_argument("--helper", choices=["none","encoder","prefill","idle-encoder"], default="none")
    p.add_argument("--partition", choices=["position","visual_first","visual_all","matched_visual_count"], default="position")
    p.add_argument("--ratio", type=float, default=.5)
    p.add_argument("--requests", type=int, default=128)
    p.add_argument("--arrival", type=Path, help="Reuse a frozen stock HTTP-client arrival trace verbatim")
    p.add_argument("--qps", type=float, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=20261127014)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--attention-engine", choices=["eager", "graph"], default="eager")
    p.add_argument("--vision-backend", choices=["stock-sdpa", "packed-fa3"], default="stock-sdpa")
    p.add_argument("--encoder-pipeline", action="store_true")
    p.add_argument("--graph-bundles", type=int, default=64)
    p.add_argument("--graph-context", type=int, choices=(8192, 16384, 32768, 65536), default=32768)
    p.add_argument("--d-gib", type=float, default=8.)
    p.add_argument("--helper-gib", type=float, default=8.)
    p.add_argument("--output", type=int, help="Fixed generation length. Default Natural EOS up to 2048.")
    p.add_argument("--instrument", action="store_true")
    args = p.parse_args()
    if args.workload == "realistic" and (args.family != "qwen-vl" or args.topology == "dual-prefill"):
        p.error("Real image runs require either E/P/D or a vision-and-prefill worker with two decoders")
    if args.helper in ("encoder","idle-encoder") and args.topology != "epd":
        p.error("Only EPD assigns a dedicated encoder role")
    if args.helper != "none" and args.topology == "dual-decode":
        p.error("The same-budget dual decode layout is fully local")
    if args.output is not None and not 16 <= args.output <= 256:
        p.error("Fixed performance outputs must be 16–256 tokens")
    if not 0. <= args.ratio <= 1. or min(args.d_gib, args.helper_gib, args.max_batch) <= 0:
        p.error("Specify positive capacities and a ratio from zero to one")
    if torch.cuda.device_count() != 3:
        p.error("Lease exactly three H200s")
    rows = checked_rows(args)
    arrival_plan = None
    if args.arrival is not None:
        arrival = rt.read(args.arrival)
        if args.qps != [arrival["QPS"]] or arrival["serial"] or len(arrival["trace"]) < args.requests:
            p.error("Replay requires the same QPS and the requested number of stock open-loop arrivals")
        roster = {row["request_id"]: row for row in rows}
        arrival_plan = [(roster[item["request_id"]], item["offset_seconds"])
                        for item in arrival["trace"][:args.requests]]
        args.seed = arrival["seed"]
    args.out.mkdir()
    rt.write(args.out / "config.json", vars(args) | {k:str(getattr(args,k)) for k in ("model","manifest","out")}
             | {"manifest_sha256":rt.sha(args.manifest), "config_sha256":rt.sha(args.model/"config.json"),
                "arrival": str(args.arrival) if args.arrival else None,
                "arrival_sha256": rt.sha(args.arrival) if args.arrival else None,
                "complete_roster_ids":[r["request_id"] for r in rows],
                "source_sha256": {file:rt.sha(Path(__file__).with_name(file))
                                  for file in ("online.py","model.py","cache.py","peer_attention.py","paged_graph.py")}})
    with torch.inference_mode():
        session = Session(args)
        warm = session.cohort(sorted(rows,key=lambda x: x["total_input_tokens"])[:1],1,1.,args.seed,
                              args.output or 16, warming=True)
        rt.write(args.out / "warmup.json", warm)
        for i,qps in enumerate(args.qps):
            result = session.cohort(rows,args.requests,qps,args.seed,args.output or 2048,arrival_plan=arrival_plan)
            rt.write(args.out / f"cohort_{i+1}.json", result)
            print(args.family,args.topology,args.helper,qps,result["outcomes"],result["latencies"],
                  result["goodput_requests_per_second"],flush=True)
        for pool in session.graphs.values():
            pool.close()


if __name__ == "__main__":
    main()
