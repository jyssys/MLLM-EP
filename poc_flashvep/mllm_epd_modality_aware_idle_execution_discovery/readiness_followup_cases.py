"""Freeze ordered, image-complete causal prefixes for a matched P/D follow-up."""

import argparse
from hashlib import file_digest, sha256
import io
import json
from pathlib import Path

import numpy as np


REQUEST_PREFIX = "yubo2333__MMLongBench-Doc__"
CASES = (
    ("small_1152", "174", 50, 2048),
    ("small_1721", "965", 50, 2048),
    ("medium_4084", "725", 50, 4096),
    ("large_6842", "178", 50, 8192),
    ("late_7739", "170", 25, 8192),
    ("no_proven_lead_7739", "168", 0, 8192),
)
SAME_DOC_SHORTER = (
    ("same_doc_first_image", "725", 50, None),
    ("same_doc_le1024", "725", 50, 1024),
    CASES[2],
)
VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655


def digest(raw):
    return sha256(raw).hexdigest()


def verify(value, description):
    if not value:
        raise ValueError(description)


def read_original(row):
    data_bytes = Path(row["token_metadata_file"]).read_bytes()
    verify(digest(data_bytes) == row["token_metadata_SHA256"],
           f"Processor artifact changed for {row['request_id']}")
    with np.load(io.BytesIO(data_bytes), allow_pickle=False) as data:
        tokens = data["input_ids"].reshape(-1)
        positions = data["positions"]
        grids = data["image_grid_thw"].tolist()
    verify(len(tokens) == row["total_input_tokens"] and positions.shape == (3, 1, len(tokens))
           and grids == row["grid_thw"], f"Processor geometry differs for {row['request_id']}")
    return tokens, positions


def fixture(source_rows, historical, case_id, source_id, lead_ms, row_cap):
    request_id = REQUEST_PREFIX + source_id
    row = source_rows[request_id]
    evidence = historical[request_id]
    spans = row["image_visual_spans"]
    verify(len(spans) == len(row["images"]) == row["image_count"] == evidence["total_images"]
           and row["total_input_tokens"] == evidence["LM_input_rows"],
           f"Observed image or prompt count differs for {request_id}")
    tokens, positions = read_original(row)
    cutoff_sizes = []
    previous_end = 0
    for (start, end), image in zip(spans, row["images"], strict=True):
        verify(previous_end < start < end < len(tokens) and image["visual_span"] == [start, end]
               and tokens[start - 1] == VISION_START and tokens[end] == VISION_END
               and np.all(tokens[start:end] == IMAGE_PAD), f"Invalid original contiguous image span for {request_id}")
        cutoff_sizes.append(end + 1)
        previous_end = end

    lead_options = evidence["max_complete_contiguous_prefix_by_lead_ms"]
    if lead_ms:
        last_early = lead_options[str(lead_ms)]
        maximum_images = last_early["images"]
        verify(0 < maximum_images <= len(spans) and cutoff_sizes[maximum_images - 1] == last_early["LM_rows"],
               f"Historical timing frontier differs from original delimiters for {request_id}")
    else:
        verify(evidence["first_image_lead_lower_ms"] == 0, f"Expected zero demonstrated lead for {request_id}")
        maximum_images = len(spans)
    if row_cap is None:
        row_cap = cutoff_sizes[0]
    prefix_images, cut = max((i + 1, end) for i, end in enumerate(cutoff_sizes[:maximum_images]) if end <= row_cap)
    verify(prefix_images < len(spans), f"No remaining image after {request_id} cut")
    return {
        "case_id": case_id,
        "request_id": request_id,
        "cut_tokens": cut,
        "prefix_images": prefix_images,
        "total_tokens": len(tokens),
        "total_images": len(spans),
        "historical_lead_lower_bound_ms": lead_ms,
        "historical_first_image_lead_lower_ms": evidence["first_image_lead_lower_ms"],
        "arms": ["native", "late_p", "early_p", "early_d"],
        "prefix_token_ids_SHA256_int64_le": digest(tokens[:cut].astype("<i8", copy=False).tobytes()),
        "prefix_positions_SHA256_int64_le": digest(positions[..., :cut].astype("<i8", copy=False).tobytes()),
        "prefix_image_SHA256_in_order": [image["SHA256"] for image in row["images"][:prefix_images]],
        "token_metadata_SHA256": row["token_metadata_SHA256"],
        "prompt_SHA256": row["prompt_SHA256"],
        "request": row,
    }


def check_media(fixtures):
    verified = {}
    for case in fixtures:
        for image in case["request"]["images"]:
            path = Path(image["path"]).resolve(strict=True)
            if path not in verified:
                with path.open("rb") as stream:
                    verified[path] = file_digest(stream, "sha256").hexdigest()
            verify(verified[path] == image["SHA256"], f"Frozen original image changed at {path}")
    return len(verified)


def select(manifest, readiness, same_doc_shorter=False):
    manifest_bytes = manifest.read_bytes()
    readiness_bytes = readiness.read_bytes()
    trace = json.loads(readiness_bytes)
    source_sha = digest(manifest_bytes)
    verify(source_sha == trace["input_SHA256"]["primary_very_long_n24.json"],
           "The archived readiness trace and supplied natural manifest differ")
    source_rows = {row["request_id"]: row for row in json.loads(manifest_bytes)["requests"]}
    historical = {row["request_id"]: row for row in trace["requests"]}
    verify(source_rows.keys() == historical.keys() and len(source_rows) == trace["requests_validated"] == 24,
           "The archived trace and natural request rosters differ")
    fixtures = [fixture(source_rows, historical, *setting)
                for setting in (SAME_DOC_SHORTER if same_doc_shorter else CASES)]
    if same_doc_shorter:
        unique = {}
        for case in fixtures:
            unique.setdefault(case["cut_tokens"], case)
        fixtures = list(unique.values())
    else:
        lead_case, control = fixtures[-2:]
        for key in ("cut_tokens", "prefix_images", "total_images", "total_tokens", "prefix_image_SHA256_in_order",
                    "prefix_token_ids_SHA256_int64_le", "prefix_positions_SHA256_int64_le"):
            verify(lead_case[key] == control[key], f"No-lead matched control differs on {key}")
    return {
        "evidence": "CPU_VERIFIED_PROSPECTIVE_FIXTURE_SELECTION",
        "scope": "Original image-complete prefixes selected with prior diagnostic IPC bounds. These data do not predict fresh live readiness, opportunity frequency, or model latency.",
        "manifest_SHA256": source_sha,
        "readiness_receipt_SHA256": digest(readiness_bytes),
        "source_media_files_validated": check_media(fixtures),
        "cases": fixtures,
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compact", type=Path)
    parser.add_argument("--same-doc-shorter", action="store_true", help="Vary the prefix size on original request 725")
    args = parser.parse_args()
    result = select(args.manifest, args.readiness, args.same_doc_shorter)
    write(args.out, result)
    if args.compact:
        cases = [{key: value for key, value in row.items() if key != "request"} for row in result["cases"]]
        write(args.compact, result | {"cases": cases})
    for row in result["cases"]:
        print(row["case_id"], row["request_id"], row["cut_tokens"], row["prefix_images"], row["historical_lead_lower_bound_ms"])


if __name__ == "__main__":
    main()
