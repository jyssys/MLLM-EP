"""Natural-generation open-loop loads with all offered requests retained."""

import json
import numpy as np
import torch

from .common import ROOT, write
from .worker import new_engine, close, serve, guard


def pick_cases(split, count, seed):
    data = json.loads((ROOT / "traffic_workload_manifest.json").read_text())
    rng = np.random.default_rng(seed)
    chat = [
        r
        for r in data["cases"]
        if r["split"] == split and r["domain"] == "conversation_proxy"
    ]
    p2 = [
        r for r in data["cases"] if r["split"] == split and r.get("prompt_bin") == 2048
    ]
    p8 = [
        r for r in data["cases"] if r["split"] == split and r.get("prompt_bin") == 8192
    ]
    assert chat and p2 and p8
    output = []
    # Frozen shape mixture: 50% conversation, 25% article2K, 25% article8K.
    for i in range(count):
        choices = (chat, chat, p2, p8)[i % 4]
        output.append(choices[int(rng.integers(len(choices)))])
    rng.shuffle(output)
    return output


def main_job(model, job, target):
    manifest = json.loads((ROOT / "traffic_workload_manifest.json").read_text())
    cells = job.get("cells", [["atomic", 1024]])
    results = []
    for mode, chunk in cells:
        marker = target / f"{mode}_C{chunk}_complete.json"
        if marker.exists():
            continue
        width = job.get("width", 8)
        budget = 16384 if mode == "no_chunk" else 1024
        native_mode = "atomic" if mode == "no_chunk" else mode
        actual_chunk = 16384 if mode == "no_chunk" else chunk
        engine = new_engine(model, native_mode, actual_chunk, width, budget)
        if mode == "no_chunk":
            from dataclasses import replace

            original_take = engine.scheduler.take_phases

            def one_whole_prompt(
                phases, _scheduler=engine.scheduler, _take=original_take, **kwargs
            ):
                saved = _scheduler.config
                if any(phase.value == "prefill" for phase in phases):
                    _scheduler.config = replace(saved, max_num_seqs=1)
                try:
                    return _take(phases, **kwargs)
                finally:
                    _scheduler.config = saved

            engine.scheduler.take_phases = one_whole_prompt
        engine.frontend_stop_token_ids = manifest["frontend_stop_token_ids"]
        warm_cases = pick_cases("calibration", width, 20261023)
        warm = serve(
            engine,
            warm_cases,
            job.get("warmup_cap", 32),
            [0] * width,
            0.001,
            observe=False,
        )
        write(target / f"warm_{mode}_C{chunk}.json", warm)
        for repetition in job.get("trials", [0]):
            if job.get("pilot"):
                cases = pick_cases(
                    "calibration", job.get("requests", 8), 300 + repetition
                )
                schedule = [0] * len(cases)
                duration = 0.001
            else:
                duration = job.get("duration_s", 40.0)
                rate = job["rate"]
                rng = np.random.default_rng(job["arrival_seed"] + repetition)
                if job.get("pattern", "poisson") == "poisson":
                    offsets = np.cumsum(
                        rng.exponential(
                            1 / rate, size=max(100, int(rate * duration * 4))
                        )
                    )
                    schedule = [float(x) for x in offsets if x < duration]
                else:
                    schedule = []
                    for offset in np.arange(0, duration, 4 / rate):
                        schedule.extend([float(offset)] * 4)
                cases = pick_cases(
                    job.get("split", "heldout"),
                    len(schedule),
                    job.get("input_seed", 20261024) + repetition,
                )
            file = target / f"{mode}_C{chunk}_T{repetition}.json"
            if file.exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            result = serve(
                engine,
                cases,
                job.get("cap", 512),
                schedule,
                duration,
                drain=job.get("drain_s", 120),
                trace=job.get("trace", False),
                observe=not job.get("observer_off", False),
            )
            result.update(
                job=job,
                policy=mode,
                C=chunk,
                G=4
                if mode in ("atomic", "staged4", "no_chunk")
                else 2
                if mode == "grouped"
                else 1,
                GPU=guard(),
                repetition=repetition,
                rate=job.get("rate"),
                generation_cap=job.get("cap", 512),
                pattern=job.get("pattern", "burst_pilot"),
                no_chunk_token_budget_exception=budget if mode == "no_chunk" else None,
            )
            write(file, result)
            results.append(
                dict(
                    policy=mode,
                    offered=len(cases),
                    EOS_or_stop=sum(
                        r.get("finish_reason") in ("stop", "frontend_stop")
                        for r in result["requests"]
                    ),
                    capped=sum(
                        r.get("finish_reason") == "length" for r in result["requests"]
                    ),
                    errors=sum(r["status"] != "COMPLETE" for r in result["requests"]),
                )
            )
            print(
                "TRAFFIC",
                job["id"],
                mode,
                chunk,
                repetition,
                results[-1],
                "wall",
                round(result["end"] - result["start"], 2),
                flush=True,
            )
        write(marker, dict(status="COMPLETE"))
        if mode == "no_chunk":
            engine.scheduler.take_phases = original_take
            del one_whole_prompt
            del original_take
        close(engine)
        del engine
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return dict(results=results)
