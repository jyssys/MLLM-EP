"""Natural decode-only regression control, no prefill interference."""

import gc
import json
import time
import torch
from .common import ROOT, MODEL, configuration, guard, write


@torch.inference_mode()
def main():
    from vllm_rlt import SamplingParams
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.models import OuroForCausalLM

    guard()
    torch.set_num_threads(1)
    model = OuroForCausalLM.from_pretrained(
        MODEL, device="cuda:0", dtype=torch.bfloat16
    )
    manifest = json.loads((ROOT / "workload_manifest.json").read_text())
    output = []
    for concurrency in (1, 4):
        for mode in ("atomic", "grouped"):
            e = StageEngine(model, **configuration(mode, 1024, concurrency))
            cases = [
                r
                for r in manifest["cases"]
                if r["trial"] == 2 and "background_index" in r
            ][:concurrency]
            times = {r["id"]: [] for r in cases}
            tokens = {}
            for r in cases:
                e.add_request(
                    r["id"], r["input_ids"], SamplingParams(max_tokens=128, max_loops=4)
                )
            origin = None
            while e.has_unfinished_requests():
                for r in e.step():
                    times[r.request_id].append(time.perf_counter())
                    tokens[r.request_id] = r.token_ids
                if origin is None and all(times.values()):
                    origin = time.perf_counter()
            e.model_runner.synchronize()
            output.append(
                dict(
                    mode=mode,
                    concurrency=concurrency,
                    evidence="LIVE_MEASURED_CONTROL",
                    decode_span_s=time.perf_counter() - origin,
                    ITLs_s={
                        rid: [b - a for a, b in zip(v, v[1:]) if b >= origin]
                        for rid, v in times.items()
                    },
                    tokens=tokens,
                    backend=e.cache_manager.backend,
                )
            )
            print(
                "DECODE_CONTROL",
                mode,
                concurrency,
                output[-1]["decode_span_s"],
                flush=True,
            )
            e.close()
            del e
            gc.collect()
            torch.cuda.empty_cache()
    write(ROOT / "decode_only_control.json", output)


if __name__ == "__main__":
    main()
