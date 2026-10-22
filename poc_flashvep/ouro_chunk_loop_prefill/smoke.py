"""Pretrained GPU4 Atomic/Split/Mixed correctness. Diagnostic timing only."""

import argparse
import gc
import time
import torch
from transformers import AutoTokenizer
from .common import ROOT, MODEL, guard, configuration, write


def differences(reference, candidate):
    a, b = reference.float(), candidate.float()
    return dict(
        exact=torch.equal(reference, candidate),
        max_abs=float((a - b).abs().max()),
        rel_l2=float(
            torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(a).clamp_min(1e-12)
        ),
    )


@torch.inference_mode()
def probe(model, ids, mode, chunk, streams, graphs):
    from vllm_rlt import SamplingParams
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase

    blocks = max(256, 4 * ((len(ids) + 24) // 16))
    engine = StageEngine(
        model,
        **configuration(
            mode, chunk, concurrency=2, blocks=blocks, streams=streams, graphs=graphs
        ),
    )
    trace = []
    engine.add_request(
        "probe", ids, SamplingParams(max_tokens=8, max_loops=4, exit_threshold=1.0)
    )
    hidden = logits = None
    kv = []
    started = time.perf_counter()
    while engine.has_unfinished_requests():
        request = engine.scheduler.requests.get("probe")
        if request and request.phase == RequestPhase.CODA and hidden is None:
            engine.model_runner.synchronize()
            hidden = request.hidden_state.clone().cpu()
            logits = model.coda(request.hidden_state.reshape(1, -1)).cpu()
            for depth in range(4):
                for layer in range(24):
                    k, v = engine.cache_manager.read(
                        layer, "probe", depth, length=len(ids)
                    )
                    kv.extend([k.cpu(), v.cpu()])
        outputs = engine.step()
        batch = engine.last_schedule
        trace.append(
            dict(
                phases=[batch.phase_for(r).value for r in batch.requests],
                spans=[
                    dict(start=r.token_start, count=r.token_count, depth=r.depth)
                    for r in batch.requests
                ],
            )
        )
        for output in outputs:
            final = output
    engine.model_runner.synchronize()
    result = dict(
        tokens=list(final.token_ids),
        depths=list(final.exit_depths),
        reason=final.finish_reason,
        backend=engine.cache_manager.backend,
        backend_info=engine.cache_manager.attention_info,
        diagnostic_wall_s=time.perf_counter() - started,
        trace=trace,
        allocated=torch.cuda.memory_allocated(),
        peak=torch.cuda.max_memory_allocated(),
        graphs=graphs,
        streams=streams,
    )
    engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result, (hidden, logits, kv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", default="32,128,512")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--no-streams", action="store_true")
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--label", required=True)
    p.add_argument("--modes", default="atomic,split,mixed,grouped")
    a = p.parse_args()
    guard()
    torch.set_num_threads(1)
    from vllm_rlt.models import OuroForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=True
    )
    model = OuroForCausalLM.from_pretrained(
        MODEL, device="cuda:0", dtype=torch.bfloat16
    )
    base = tokenizer.encode(
        "Read the following mathematical explanation and summarize its conclusion. "
        "A rectangle has length twelve and width seven. Its area is eighty-four. "
        * 1000
    )
    folder = ROOT / "raw" / a.label
    folder.mkdir(parents=True, exist_ok=False)
    for length in map(int, a.lengths.split(",")):
        ids = base[:length]
        assert len(ids) == length
        ref = None
        for mode in a.modes.split(","):
            torch.cuda.reset_peak_memory_stats()
            print("PROBE", length, mode, flush=True)
            try:
                row, tensors = probe(
                    model, ids, mode, a.chunk, not a.no_streams, not a.no_graphs
                )
                row.update(mode=mode, prompt_tokens=length, chunk_tokens=a.chunk)
                if ref is None:
                    ref = (row, tensors)
                else:
                    r, t = ref
                    row["parity"] = dict(
                        tokens=row["tokens"] == r["tokens"],
                        depths=row["depths"] == r["depths"],
                        hidden=differences(t[0], tensors[0]),
                        logits=differences(t[1], tensors[1]),
                        kv_max_rel_l2=max(
                            differences(x, y)["rel_l2"]
                            for x, y in zip(t[2], tensors[2])
                        ),
                        kv_all_exact=all(
                            torch.equal(x, y) for x, y in zip(t[2], tensors[2])
                        ),
                    )
                write(folder / f"P{length}_{mode}.json", row)
                print("RESULT", length, mode, row.get("parity"), flush=True)
                del tensors
            except Exception as error:
                import traceback

                write(
                    folder / f"P{length}_{mode}_FAIL.json",
                    dict(error=repr(error), traceback=traceback.format_exc()),
                )
                raise


if __name__ == "__main__":
    main()
