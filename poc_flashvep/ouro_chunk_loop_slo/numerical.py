"""Pretrained component probes and diagnostic batch-geometry replay only."""

import gc
import torch
from transformers import AutoTokenizer
from .common import MODEL, write


@torch.inference_mode()
def main_job(model, job, target):
    from poc_flashvep.ouro_chunk_loop_prefill.smoke import probe, differences

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, local_files_only=True
    )
    ids = tokenizer.encode(
        "Read the following mathematical explanation and summarize its conclusion. A rectangle has length twelve and width seven. Its area is eighty-four. "
        * 1000
    )
    comparisons = 0
    for length in (128, 512, 2048, 8192):
        ref = None
        for mode in ("atomic", "split", "grouped", "mixed"):
            row, tensors = probe(model, ids[:length], mode, job["chunk"], True, True)
            row.update(
                prompt_tokens=length,
                chunk_tokens=job["chunk"],
                mode=mode,
                scope="PRETRAINED_NUMERICAL_PROBE_NOT_ACCURACY_OR_PERFORMANCE",
            )
            if ref is None:
                ref = row, tensors
            else:
                a, t = ref
                row["parity"] = dict(
                    tokens=a["tokens"] == row["tokens"],
                    depths=a["depths"] == row["depths"],
                    hidden=differences(t[0], tensors[0]),
                    logits=differences(t[1], tensors[1]),
                    KV_exact=all(torch.equal(x, y) for x, y in zip(t[2], tensors[2])),
                    KV_max_relative_L2=max(
                        differences(x, y)["rel_l2"] for x, y in zip(t[2], tensors[2])
                    ),
                )
                comparisons += 1
            write(target / f"P{length}_C{job['chunk']}_{mode}.json", row)
            print("NUMERIC", length, job["chunk"], mode, row.get("parity"), flush=True)
            del tensors
        del ref
        gc.collect()
    return dict(numerical_comparisons=comparisons)


@torch.inference_mode()
def geometry_job(model, job, target):
    from .worker import new_engine, close
    from poc_flashvep.ouro_chunk_loop_prefill.bench import run
    from poc_flashvep.ouro_chunk_loop_prefill.smoke import differences

    collected = []
    row_hidden = []
    for mode in ("split", "grouped"):
        e = new_engine(model, mode, 1024, 8)
        run(model, mode, 1024, 8192, 8, 0, False, True, False, e)
        original_submit = e.executor.submit
        original_forward = e.model_runner.runtime.forward
        state = dict(ids=[], count=0, captured=None)

        def submit(batch, _state=state, _call=original_submit):
            _state["ids"] = [r.request.request_id for r in batch.requests]
            return _call(batch)

        def forward(batch, _state=state, _call=original_forward):
            result = _call(batch)
            if batch.operation.value == "coda" and "bg_11_3" in _state["ids"]:
                if _state["count"] == 8:
                    index = _state["ids"].index("bg_11_3")
                    # Native stream dependencies already order this consumer.
                    _state["captured"] = (
                        batch.hidden[index].clone(),
                        result.logits[index].clone(),
                        len(_state["ids"]),
                        index,
                    )
                _state["count"] += 1
            return result

        e.executor.submit = submit
        e.model_runner.runtime.forward = forward
        data = run(model, mode, 1024, 8192, 8, 11, False, True, True, e)
        e.model_runner.synchronize()
        e.executor.submit = original_submit
        e.model_runner.runtime.forward = original_forward
        captured = state["captured"]
        assert captured is not None
        hidden, original_logits, width, index = captured
        reference = torch.nn.functional.linear(
            hidden.float().reshape(1, -1), model.lm_head.weight.float()
        )[0]
        geometry = []
        for rows in (1, 2, 4, 7, 8):
            output = torch.nn.functional.linear(
                hidden.reshape(1, -1).repeat(rows, 1), model.lm_head.weight
            )
            comparison = differences(reference, output[0])
            top2 = torch.topk(output[0].float(), 2)
            geometry.append(
                dict(
                    rows=rows,
                    top1=int(top2.indices[0]),
                    top1_definition="First topk index. Not necessarily greedy argmax when logits tie.",
                    greedy_argmax=int(output[0].argmax()),
                    top1_margin=float(top2.values[0] - top2.values[1]),
                    relative_L2_to_FP32=comparison["rel_l2"],
                    max_abs_to_FP32=comparison["max_abs"],
                    same_argmax_as_FP32=int(output[0].argmax())
                    == int(reference.argmax()),
                )
            )
        tokens = next(r["output_ids"] for r in data["requests"] if r["id"] == "bg_11_3")
        collected.append(
            dict(
                mode=mode,
                captured_output_index=8,
                native_coda_actual_rows=width,
                target_row=index,
                output_prefix=tokens[:9],
                original_argmax=int(original_logits.argmax()),
                FP32_argmax=int(reference.argmax()),
                same_hidden_batch_geometry=geometry,
                evidence="DIAGNOSTIC_LM_HEAD_REPLAY_NOT_FREE_RUNNING_QUALITY_OR_METHOD_SPEEDUP",
            )
        )
        row_hidden.append(hidden.cpu())
        write(target / f"free_running_{mode}.json", data)
        del hidden, original_logits, reference, captured
        state["captured"] = None
        del original_forward, original_submit, submit, forward, state
        close(e)
        del e
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(
        probes=collected,
        hidden_difference=differences(row_hidden[0], row_hidden[1]),
        same_prefix_before_capture=collected[0]["output_prefix"][:8]
        == collected[1]["output_prefix"][:8],
        limitation="A batch-shape logit difference does not prove the cause of every old mismatch. No forced token enters headline serving.",
    )
    write(target / "batch_geometry_diagnostic.json", result)
    return dict(status="GEOMETRY_DIAGNOSTIC_COMPLETE")
