"""Freeze compatible, dataset-stratified subsets of the existing Discovery200."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SEED = "h200-architecture-20261127-v1"
PROFILE_SIZE = 24
SMOKE_SIZE = 6
RULES = {
    "cogvlm2": {"maximum_images": 1, "scope": "The published conversation builder accepts one intact image."},
    "idefics": {"maximum_images": 16, "scope": "Study resource envelope for initial profile, not a model limit."},
}


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def bucket(row):
    images = row["image_count"]
    return "01" if images == 1 else "02–04" if images <= 4 else "05–08" if images <= 8 else "09–16"


def choose(rows, limit, family):
    groups = defaultdict(list)
    for row in rows:
        key = row["source"] if family == "cogvlm2" else (bucket(row), row["study_source"])
        groups[key].append(row)
    for group in groups.values():
        group.sort(key=lambda row: hashlib.sha256(f"{SEED}|{family}|{row['request_id']}".encode()).digest())
    picks = []
    while len(picks) < min(len(rows), limit):
        for key in sorted(groups):
            if groups[key] and len(picks) < limit:
                picks.append(groups[key].pop(0))
    return picks


def publish(path, value):
    body = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text() != body:
            raise ValueError(f"Refusing to overwrite frozen cohort {path}")
    else:
        path.write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery200", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.discovery200.read_text())
    rows = original["requests"]
    if len(rows) != 200 or len({row["request_id"] for row in rows}) != 200:
        raise ValueError("Use the frozen grouped-v2 Discovery200")
    if set(original["discovery_ids"]).intersection(original["validation_ids"]):
        raise ValueError("The frozen split overlaps")
    registry = {"seed": SEED, "source_sha256": sha(args.discovery200), "families": {}}
    for family, rule in RULES.items():
        compatible = [row for row in rows if 0 < row["image_count"] <= rule["maximum_images"]]
        discovery = [row for row in compatible if row["study_split"] == "discovery"]
        validation = [row for row in compatible if row["study_split"] == "validation"]
        profile = choose(discovery, PROFILE_SIZE, family)
        smoke = profile[:SMOKE_SIZE]
        for label, roster in (("discovery", discovery), ("validation", validation),
                              ("profile", profile), ("smoke", smoke)):
            publish(args.out / f"{family}_{label}.json", {"name": f"{family}_{label}", "family": family,
                                                          "source_sha256": registry["source_sha256"],
                                                          "requests": roster, "seed": SEED, **rule})
        registry["families"][family] = {
            **rule, "discovery_eligible_by_source": dict(Counter(row["source"] for row in discovery)),
            "validation_eligible_by_source": dict(Counter(row["source"] for row in validation)),
            "excluded": [{"request_id": row["request_id"], "image_count": row["image_count"],
                          "reason": rule["scope"]} for row in rows if row not in compatible],
            "profile": [{key: row[key] for key in ("request_id", "source", "image_count", "prompt_SHA256")}
                        for row in profile],
            "smoke": [row["request_id"] for row in smoke],
            "context_qualification": "Candidate processor still measures actual input lengths. "
                                     "Matching and context budget use the original unmodified prompt.",
        }
    publish(args.registry, registry)


if __name__ == "__main__":
    main()
