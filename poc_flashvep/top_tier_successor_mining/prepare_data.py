"""Download public calibration/evaluation data with explicit provenance.

Never substitutes generated images or self-generated labels. Calibration and
evaluation records are image-disjoint for GQA. This script does not use CUDA.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    api = HfApi(token=False)
    wanted = {
        "lmms-lab/GQA": ["testdev_balanced_images/testdev-00000-of-00001.parquet",
                         "testdev_balanced_instructions/testdev-00000-of-00001.parquet"],
        "HuggingFaceM4/ChartQA": ["data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet",
                                 "data/val-00000-of-00001-0f11003c77497969.parquet"],
    }
    files, sources = {}, []
    jobs = []
    for repo, filenames in wanted.items():
        sha = api.dataset_info(repo).sha
        for filename in filenames:
            jobs.append((repo, filename, sha))

    def download(job):
        repo, filename, sha = job
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=sha,
                               token=False, local_dir=root / "downloads" / repo)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return job, path, digest

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for job, path, digest in pool.map(download, jobs):
            repo, filename, sha = job
            files[filename] = path
            sources.append(dict(repo=repo, filename=filename, revision=sha, sha256=digest))
            print(json.dumps(sources[-1]), flush=True)
    (root / "sources.json").write_text(json.dumps(sources, indent=2))
    imgfile, qfile = wanted["lmms-lab/GQA"]
    images = pq.read_table(files[imgfile]).to_pylist()
    questions = pq.read_table(files[qfile]).to_pylist()
    print("GQA schemas", list(images[0]), list(questions[0]), flush=True)
    image_map = {}
    for row in images:
        identity = str(row["id"])
        image = row["image"]
        if image.get("bytes") is None:
            raise ValueError("GQA download did not embed image bytes")
        out = root / "images" / f"gqa_{identity}.png"
        Image.open(io.BytesIO(image["bytes"])).convert("RGB").save(out)
        image_map[identity] = str(out.resolve())
    rng = np.random.default_rng(73091)
    ids = sorted(image_map)
    rng.shuffle(ids)
    calib_images = set(ids[:len(ids)//2])
    records = []
    for i, row in enumerate(questions):
        iid = str(row["imageId"])
        if iid not in image_map:
            continue
        split = "calibration" if iid in calib_images else "heldout"
        records.append({"id": f"gqa_{i}", "dataset": "gqa", "split": split,
                        "image_id": iid, "images": [image_map[iid]],
                        "question": row["question"], "answers": [row["answer"]],
                        "full_answer": row.get("fullAnswer", row["answer"]),
                        "source_row": i})
    for fname in wanted["HuggingFaceM4/ChartQA"]:
        rows = pq.read_table(files[fname]).to_pylist()
        split = "heldout" if "/test-" in fname else "calibration_chart_control"
        print("ChartQA schema", list(rows[0]), flush=True)
        for i, row in enumerate(rows):
            image_bytes = row["image"]["bytes"]
            digest = hashlib.sha256(image_bytes).hexdigest()[:20]
            out = root / "images" / f"chartqa_{digest}.png"
            if not out.exists():
                Image.open(io.BytesIO(image_bytes)).convert("RGB").save(out)
            records.append({"id": f"chartqa_{split}_{i}", "dataset": "chartqa", "split": split,
                            "image_id": digest, "images": [str(out.resolve())],
                            "question": row["query"], "answers": row["label"],
                            "full_answer": row["label"][0], "source_row": i,
                            "human_or_machine": row.get("human_or_machine")})
    (root / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    # FineWeb-Edu public dataset-server API avoids downloading a multi-GB shard
    # for the paper's 400 calibration sequences. Preserve the raw responses.
    text_records = []
    for offset in range(0, 800, 100):
        params = dict(dataset="HuggingFaceFW/fineweb-edu", config="sample-10BT",
                      split="train", offset=offset, length=100)
        response = requests.get("https://datasets-server.huggingface.co/rows", params=params, timeout=90)
        response.raise_for_status()
        raw = response.json()
        (root / f"fineweb_rows_{offset}.json").write_text(json.dumps(raw))
        for r in raw["rows"]:
            text_records.append({"id": f"fineweb_{r['row_idx']}", "text": r["row"]["text"],
                                 "source_row": r["row_idx"]})
    (root / "fineweb.jsonl").write_text("".join(json.dumps(r) + "\n" for r in text_records))
    summary = {"requests": len(records), "fineweb": len(text_records), "seed": 73091,
               "gqa_calibration_images": len(calib_images), "sources": sources}
    (root / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
