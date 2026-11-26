"""Verify actual D clone ranges and copy-before-ACK using worker records."""

import json


def jsonl(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def clonelist(raw, image_keys, spans, cut):
    expected = {}
    for key, (begin, end) in zip(image_keys, spans, strict=True):
        if end > cut:
            expected[key] = (max(0, cut - begin), end - max(begin, cut))
    clones = []
    for path in raw.glob("ec_cuda_ipc_decode_gpu*.jsonl"):
        for entry in jsonl(path):
            for clone in entry.get("clone_shapes", []):
                if clone["key"] not in expected:
                    continue
                first, count = expected[clone["key"]]
                clones.append({
                    **clone, "expected_first_source_row": first, "expected_row_count": count,
                    "copy_before_ACK": entry["clone_complete_before_ACK"],
                })
    valid = len(expected) == len(clones) and set(expected) == {entry["key"] for entry in clones} and all(
        clone["first_source_row"] == clone["expected_first_source_row"]
        and clone["clone_shape"][0] == clone["expected_row_count"] and clone["copy_before_ACK"]
        for clone in clones
    )
    return {"verified_range": valid, "clones": clones,
            "strictly_smaller_clones": sum(item["clone_shape"][0] < item["source_shape"][0] for item in clones)}
