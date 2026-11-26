"""Validate frozen image geometry and choose calibration cuts without timing data."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path


BF16_EMBEDDING_BYTES_PER_TOKEN = 3584 * 2
BLOCK_TOKENS = 128
RATIOS = (0, .05, .10, .20, .30)
QUANTA = (128, 256, 512, 1024, 2048, 4096)
# These request indices are fixed across starts. The two groups cover single,
# few-image, multi-page, and long-document prompts with distinct request IDs.
FIT_INDICES = (0, 2, 4, 7, 13, 14, 19, 21)
VALIDATION_INDICES = (1, 6, 15, 20, 23, 29, 31)
ABLATION_INDICES = (0, 1, 4, 6, 13, 15, 19, 21, 29)


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def request_geometry(row, verify_archive=False):
    """Read processor metadata. Output excludes images, text, IDs of tokens and references."""
    length = row["total_input_tokens"]
    images = row["images"]
    spans = row["image_visual_spans"]
    grids = row["grid_thw"]
    require(len(images) == row["image_count"] == len(spans) == len(grids), row["request_id"])
    visual = 0
    images_out = []
    previous = 0
    for index, (image, span, grid) in enumerate(zip(images, spans, grids, strict=True)):
        left, right = span
        require(previous <= left < right <= length, "Overlapping or invalid image spans")
        require(span == image["visual_span"] and grid == image["grid_thw"], "Image geometry mismatch")
        require(math.prod(grid) == 4 * (right - left) == 4 * image["visual_tokens"],
                "Unexpected Qwen2.5-VL 2×2 merge geometry")
        previous = right
        visual += right - left
        images_out.append({
            "image_index": index,
            "span": span,
            "visual_tokens": right - left,
            "grid_thw": grid,
            "original_resolution": image["original_resolution"],
            "processed_resolution": image["processed_resolution"],
            "BF16_tensor_payload_bytes": (right - left) * BF16_EMBEDDING_BYTES_PER_TOKEN,
        })
    require(visual == row["visual_tokens"], "Visual token count mismatch")
    require(length == visual + row["text_tokens"] + row["special_tokens"], "Token accounting mismatch")
    require(row["encoder_output_shape_structural"] == [visual, 3584], "Unexpected encoder shape")
    if verify_archive:
        import numpy as np

        path = Path(row["token_metadata_file"])
        require(sha256(path.read_bytes()).hexdigest() == row["token_metadata_SHA256"], "Token archive hash mismatch")
        with np.load(path, allow_pickle=False) as tensors:
            require(tensors["input_ids"].size == length, "Recorded token count mismatch")
            require(tensors["image_grid_thw"].tolist() == grids, "Recorded grid mismatch")
            types = tensors["mm_token_type_ids"].reshape(-1)
            expected = np.zeros(length, dtype=types.dtype)
            for left, right in spans:
                expected[left:right] = 1
            require(np.array_equal(types, expected), "Recorded modality mask mismatch")
    return {
        "request_id": row["request_id"],
        "prompt_sha256": row["prompt_SHA256"],
        "token_metadata_sha256": row["token_metadata_SHA256"],
        "source": row["source"],
        "stratum": row["stratum"],
        "input_tokens": length,
        "visual_tokens": visual,
        "text_tokens": row["text_tokens"],
        "special_tokens": row["special_tokens"],
        "image_count": len(images_out),
        "images": images_out,
        "frozen_output_tokens": row["natural_output_tokens"],
    }


def cross_images(geometry, prefix):
    return [image for image in geometry["images"] if image["span"][0] < prefix < image["span"][1]]


def cut_features(geometry, prefix):
    length = geometry["input_tokens"]
    require(0 < prefix <= length, "Invalid exact prefix length")
    suffix = length - prefix
    cross = cross_images(geometry, prefix)
    return {
        "prefix_tokens": prefix,
        "suffix_tokens": suffix,
        "realized_ratio": suffix / length,
        "prefix_blocks_128": math.ceil(prefix / BLOCK_TOKENS),
        "causal_attention_pairs_per_layer": prefix * suffix + suffix * (suffix + 1) // 2,
        "cross_boundary_image_count": len(cross),
        "cross_boundary_tensor_payload_bytes": sum(image["BF16_tensor_payload_bytes"] for image in cross),
        "has_image_edge": any(prefix in image["span"] for image in geometry["images"]),
    }


def matched_cuts(geometry):
    """Choose one edge and an adjacent internal cut at the same transport block count."""
    length = geometry["input_tokens"]
    target = length * .9
    matches = []
    for image in geometry["images"]:
        left, right = image["span"]
        # One adjacent token isolates the actual image-ownership change while
        # retaining nearly identical suffix work and the same number of KV blocks.
        for edge, internal in ((left, left + 1), (right, right - 1)):
            if (0 < internal < length and left < internal < right
                    and 0 < length - edge <= min(8192, int(length * .35))
                    and math.ceil(edge / BLOCK_TOKENS) == math.ceil(internal / BLOCK_TOKENS)):
                matches.append((abs(edge - target), edge, internal))
    if not matches:
        return None
    _, edge, internal = min(matches)
    generic = min(length - 128, max(1, round(target)))
    # No image information supplies this fourth arm. Tie-break toward the lower
    # aligned block and never pad input/model geometry to create an alignment.
    block_control = max(BLOCK_TOKENS, int(target // BLOCK_TOKENS) * BLOCK_TOKENS)
    return {
        "image_boundary": edge,
        "intra_image": internal,
        "generic_10pct": generic,
        "block_aligned_10pct_control": block_control,
    }


def plan(calibration):
    cells = []
    unmatched = []
    for fit_group, indices in (("fit", FIT_INDICES), ("validation", VALIDATION_INDICES)):
        for index in indices:
            row = calibration[index]
            length = row["input_tokens"]
            for ratio in RATIOS:
                cut = round(length * (1 - ratio))
                cells.append({"index": index, "request_id": row["request_id"], "series": "cost",
                              "fit_group": fit_group, "arm": f"ratio_{ratio:g}", **cut_features(row, cut)})
            # Sparse quanta at alternating contexts supplement the nominal ratios.
            quanta = QUANTA[:3] if index % 2 == 0 else QUANTA[3:]
            for quanta_size in quanta:
                if quanta_size >= length:
                    continue
                cut = length - quanta_size
                cells.append({"index": index, "request_id": row["request_id"], "series": "cost",
                              "fit_group": fit_group, "arm": f"tokens_{quanta_size}", **cut_features(row, cut)})
    for index in ABLATION_INDICES:
        row = calibration[index]
        choices = matched_cuts(row)
        if choices is None:
            unmatched.append({"index": index, "request_id": row["request_id"],
                              "reason": "No eligible image edge with a same-block adjacent interior cut"})
            continue
        for arm, cut in choices.items():
            cells.append({"index": index, "request_id": row["request_id"], "series": "transfer",
                          "fit_group": "unused", "arm": arm, **cut_features(row, cut)})
    return {"cells": cells, "unmatched_transfer_rows": unmatched,
            "selection": "Fixed indices and deterministic processor geometry only. No performance-conditioned selection."}


def inventory(main_path, calibration_path):
    main_manifest, calibration_manifest = read(main_path), read(calibration_path)
    main = [request_geometry(row, verify_archive=True) for row in main_manifest["requests"]]
    calibration = [request_geometry(row, verify_archive=True) for row in calibration_manifest["requests"]]
    require(len(main) == 100 and len({row["request_id"] for row in main}) == 100, "Unexpected Main100 roster")
    require(len(calibration) == 32 and len({row["request_id"] for row in calibration}) == 32, "Unexpected Cal32 roster")
    calibration_rows = calibration_manifest["requests"]
    contexts = [{(calibration_rows[i]["source"], calibration_rows[i]["document_id"])
                 for i in indices if calibration_rows[i]["document_id"] is not None}
                for indices in (FIT_INDICES, VALIDATION_INDICES)]
    pixels = [{image["pixel_SHA256"] for i in indices for image in calibration_rows[i]["images"]}
              for indices in (FIT_INDICES, VALIDATION_INDICES)]
    require(contexts[0].isdisjoint(contexts[1]) and pixels[0].isdisjoint(pixels[1]),
            "Calibration fit and validation share an exact document or image")
    memberships = {name: set(main_manifest[f"{name}_ids"]) for name in ("discovery", "heldout")}
    for row in main:
        row["analysis_membership"] = [name for name, ids in memberships.items() if row["request_id"] in ids]
    for row in main + calibration:
        length = row["input_tokens"]
        row["nominal_cut_geometry"] = [{"requested_ratio": ratio, **cut_features(row, round(length * (1 - ratio)))}
                                       for ratio in RATIOS[1:]]
        edges = {edge for image in row["images"] for edge in image["span"]}
        row["image_edge_count_in_5_to_30pct_suffix_range"] = sum(.05 <= (length - edge) / length <= .30
                                                                for edge in edges)
    return {"evidence": "Validated frozen processor metadata",
            "main_manifest_sha256": sha256(Path(main_path).read_bytes()).hexdigest(),
            "calibration_manifest_sha256": sha256(Path(calibration_path).read_bytes()).hexdigest(),
            "calibration_fit_validation": {"fit_indices": FIT_INDICES, "validation_indices": VALIDATION_INDICES,
                                           "shared_source_document_ids": 0, "shared_image_pixel_hashes": 0},
            "Main100": main, "Calibration32": calibration,
            "online_features": {"E_request_ready": "OBSERVABLE from whole-request E response",
                                "image_ready_marker": "Host registry marker, not exact GPU-ready timestamp",
                                "image_level_encoder_cost": "NOT_AVAILABLE in serving runtime",
                                "image_level_GPU_ready_time": "NOT_AVAILABLE in serving runtime"},
            "plan": plan(calibration)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = inventory(args.main, args.calibration)
    write(args.output, result)
    print(f"Validated {len(result['Main100']) + len(result['Calibration32'])} requests; "
          f"planned {len(result['plan']['cells'])} cells", flush=True)


if __name__ == "__main__":
    main()
