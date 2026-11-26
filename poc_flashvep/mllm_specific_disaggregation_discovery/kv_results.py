"""Summarize actual exported pages, logical KV, and measured remote-decode costs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def kv_rows(native, manifest, config, block_size):
    text = config.get("text_config", config)
    layers = text["num_hidden_layers"]
    heads = text["num_key_value_heads"]
    dim = text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise ValueError("Current KV calculations require the frozen BF16 configuration")
    bytes_per_row = 2 * layers * heads * dim * 2
    lookup = {row["request_id"]: row for row in manifest["requests"]}
    details = []
    for datum in native:
        if datum["status"] != "MEASURED_ONLINE":
            continue
        row = lookup[datum["request_id"]]
        prompt = datum["input_tokens"]
        # Only produced tokens subsequently consumed by a decode step can add KV.
        gen = max(0, datum["output_tokens"] - 1)
        pages = datum["exported_prompt_block_ids"]
        expected_pages = (prompt + block_size - 1) // block_size
        if pages is not None and pages != expected_pages:
            raise ValueError(f"Unexpected native export page count for {row['request_id']}")
        visual_rows_by_page = [0] * expected_pages
        for span in row["image_visual_spans"]:
            start, stop = span if isinstance(span, list) else (span["start"], span["end"])
            for page in range(start // block_size, (stop - 1) // block_size + 1):
                visual_rows_by_page[page] += min(stop, (page + 1) * block_size) - max(start, page * block_size)
        if sum(visual_rows_by_page) != datum["visual_tokens"]:
            raise ValueError("Observed image spans do not match the original Visual KV count")
        capacity = expected_pages * block_size
        generated_pages = (prompt + gen + block_size - 1) // block_size
        full_visual_pages = sum(count == block_size for count in visual_rows_by_page)
        details.append({
            "request_id": datum["request_id"], "source": datum["source"], "study_source": datum["study_source"],
            "split": datum["study_split"], "images": datum["image_count"],
            "prefill_rows": prompt, "visual_rows": datum["visual_tokens"], "text_rows": datum["text_tokens"],
            "special_rows": datum["special_tokens"], "output_tokens": datum["output_tokens"],
            "possible_generated_kv_rows": gen, "bytes_per_row": bytes_per_row, "layers": layers,
            "kv_heads": heads, "head_dim": dim, "block_size": block_size,
            "observed_exported_prompt_pages": pages, "expected_prompt_pages": expected_pages,
            "expected_D_final_pages_without_preemption": generated_pages,
            "expected_D_final_waste_rows": generated_pages * block_size - prompt - gen,
            "prompt_page_waste_rows": capacity - prompt, "fully_visual_prompt_pages": full_visual_pages,
            "mixed_or_partial_visual_pages": sum(0 < count < block_size for count in visual_rows_by_page),
            "request_D_HTTP_ms": datum.get("D_HTTP_ms"),
        })
    return details


def read_attention_cases(run):
    return [read(path) for path in sorted(run.glob("case_*.json"))]


def break_even_rows(cases):
    details = []
    for case in cases:
        transfers = case["physical_p2p_initial_copy"]
        control = case["modes"]["local"]
        steps = case["teacher_steps"]
        ref_step = median(point["wall_ms"] for point in control["timing"]) / steps
        ref_initial = median(transfers["local"]["wall_ms"])
        for layout, mode in case["modes"].items():
            transfer = transfers[layout]
            selected_step = median(point["wall_ms"] for point in mode["timing"]) / steps
            selected_initial = median(transfer["wall_ms"]) + transfer["P_packing_cuda_ms"]
            delta_step = selected_step - ref_step
            saved_initial = ref_initial - selected_initial
            eligible_tokens = (saved_initial / delta_step if saved_initial > 0 and delta_step > 0 else None)
            profile = mode["diagnostic_profile"]["layers"]
            components = defaultdict(list)
            for step in profile:
                sum_by_component = defaultdict(float)
                for layer in step:
                    for name, ms in layer.items():
                        sum_by_component[name] += ms
                for name, value in sum_by_component.items():
                    components[name].append(value)
            row = {
                "request_id": case["request_id"], "prompt_rows": case["input_tokens"],
                "visual_rows": case["visual_tokens"], "native_output_tokens": case["native_output_tokens"],
                "layout": layout, "reserved_D_bytes": mode["KV"]["D_allocated_bytes"],
                "reserved_P_bytes": mode["KV"]["P_allocated_bytes"],
                "initial_direct_p2p_bytes": transfer["bytes"], "initial_direct_p2p_wall_median_ms": median(transfer["wall_ms"]),
                "packing_cuda_ms": transfer["P_packing_cuda_ms"],
                "wall_median_ms_per_full28_step": selected_step,
                "wall_delta_ms_per_full28_step": delta_step,
                "initial_saved_ms_vs_full_copy_after_packing": saved_initial,
                "break_even_decode_steps_if_positive": eligible_tokens,
                "modeled_net_saved_ms_at_native_output": saved_initial - max(0, case["native_output_tokens"] - 1) * delta_step,
                "teacher_greedy_matches": mode["teacher_greedy_matches"], "teacher_greedy_count": mode["teacher_greedy_count"],
                "logits_relative_L2": mode["logits"]["relative_l2"],
                "natural_equals_local": mode.get("natural", {}).get("equals_FA3_local"),
                "per_step_link_payload_bytes": (sum(case["per_layer_recurrent_link_bytes"].values()) * 28
                                                if layout != "local" else 0),
            }
            row.update({f"profile_{key}_ms_per_step": median(values) for key, values in components.items()})
            details.append(row)
    return details


def write_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("x") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attention", type=Path)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = kv_rows(read(args.native), read(args.manifest), read(args.model_config), args.block_size)
    write_csv(args.out / "native_kv.csv", rows)
    if args.attention:
        direct = break_even_rows(read_attention_cases(args.attention))
        write_csv(args.out / "decode_cost_and_break_even.csv", direct)


if __name__ == "__main__":
    main()
