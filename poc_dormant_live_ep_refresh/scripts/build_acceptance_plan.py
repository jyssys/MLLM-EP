#!/usr/bin/env python3
"""Build future-acceptance oracle keys from a baseline LLADA_DENOISE log."""

import argparse
import json
from pathlib import Path


PREFIX = "[LLADA_DENOISE]"


def parse_records(path: Path):
    records = []
    with path.open(errors="replace") as handle:
        for line in handle:
            location = line.find(PREFIX)
            if location < 0:
                continue
            record = json.loads(line[location + len(PREFIX):])
            if str(record.get("request_id", "")).startswith("warmup"):
                continue
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = parse_records(args.log)
    accepted = {}
    first_seen_masked = set()
    for record in records:
        masks = record["mask_before"]
        accepted_masks = record.get("accepted_mask")
        if accepted_masks is None:
            raise RuntimeError("baseline log lacks accepted_mask; use dormant runtime")
        for sequence_id, block_start, block_iteration, mask_row, accepted_row in zip(
            record["sequence_ids"], record["block_starts"],
            record["block_iterations"], masks, accepted_masks
        ):
            for position, (is_masked, is_accepted) in enumerate(
                zip(mask_row, accepted_row)
            ):
                key = f"{int(sequence_id)}:{int(block_start)}:{position}"
                if is_masked:
                    first_seen_masked.add(key)
                if is_accepted and key not in accepted:
                    accepted[key] = int(block_iteration)
    missing = sorted(first_seen_masked - set(accepted))
    if missing:
        raise RuntimeError(f"{len(missing)} masked keys never accepted; first={missing[:4]}")
    payload = {
        "source_log": str(args.log),
        "records": len(records),
        "masked_token_keys": len(first_seen_masked),
        "acceptance_iteration_by_key": accepted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("records", "masked_token_keys")}))


if __name__ == "__main__":
    main()
