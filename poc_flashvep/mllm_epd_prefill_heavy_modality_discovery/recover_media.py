"""Relocate Main400 while recovering missing originals from pinned source files."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from .datasets import digest, file_digest, write


CANONICAL = "07892cbe7193366ec422f1df82006c6e69f91cec"
MAIN400 = "poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124/data/main_manifest_template_v2.json"
MAIN400_SHA256 = "7ac7d28ae1678499bee35ec20e317760555944cac7b4ca29eb46aeb3957b9592"
ORIGINAL_ASSETS = Path("/home/esjung/MLLM-EP-github")
ORIGINAL_CACHE = Path("/home/esjung/.cache/huggingface/hub")


def source_location(row, original):
    source = Path(original).relative_to(ORIGINAL_CACHE)
    expected = ("datasets--"+row["source"].replace("/", "--"), "snapshots", row["source_revision"])
    if source.parts[:3] != expected:
        raise ValueError(f"Source revision mismatch for {row['request_id']}")
    return source, "/".join(source.parts[3:])


def original_source(row):
    return row["document"]["document"] if row["document"] else row["parquet"]


def download_sources(rows, cache, downloads):
    inputs = {}
    for row in rows:
        original = original_source(row)
        local, repo_path = source_location(row, original)
        if not (cache / local).is_file() and not (downloads / local).is_file():
            inputs[local] = (row["source"], row["source_revision"], repo_path)

    def fetch(item):
        from huggingface_hub import hf_hub_download

        local, (source, revision, filename) = item
        try:
            path = hf_hub_download(source, filename, repo_type="dataset", revision=revision,
                                   cache_dir=downloads, token=False)
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            print(f"SOURCE_READY {local}", flush=True)
            return str(local), None
        except Exception as error:
            return str(local), f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=3) as pool:
        return {name: error for name, error in pool.map(fetch, sorted(inputs.items())) if error is not None}


def media_file(item, assets, out):
    canonical = assets / Path(item["path"]).relative_to(ORIGINAL_ASSETS)
    restored = out / "media" / item["SHA256"][:2] / canonical.name
    return canonical if canonical.is_file() else restored


def save_media(item, data, path):
    if digest(data) != item["SHA256"]:
        raise ValueError(f"Rendered or extracted byte identity mismatch for {item['original_name']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def restore_pdf(group, file, assets, out):
    import pymupdf

    expected = group[0]["document"]["document_SHA256"]
    if file_digest(file) != expected:
        raise ValueError("Pinned PDF SHA256 differs from original Main400 data")
    media = group[0]["images"]
    with pymupdf.open(file) as pdf:
        if len(pdf) != len(media):
            raise ValueError("PDF page count mismatch")
        for page, item in zip(pdf, media, strict=True):
            path = media_file(item, assets, out)
            if not path.is_file():
                png = page.get_pixmap(dpi=72, colorspace=pymupdf.csRGB, alpha=False).tobytes("png")
                save_media(item, png, path)
    if any([i["SHA256"] for i in row["images"]] != [i["SHA256"] for i in media] for row in group):
        raise ValueError("Image order differs across requests from the same document")


def restore_parquet(group, file, assets, out):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(file)
    needed = defaultdict(list)
    indexes = sorted(group, key=lambda row: row["row_index"])
    base, cursor = 0, 0
    for index in range(parquet.num_row_groups):
        end = base + parquet.metadata.row_group(index).num_rows
        while cursor < len(indexes) and indexes[cursor]["row_index"] < end:
            needed[index].append((indexes[cursor]["row_index"]-base, indexes[cursor]))
            cursor += 1
        base = end
    if cursor != len(indexes):
        raise ValueError("Source parquet row exceeds the pinned shard")
    for number, rows in sorted(needed.items()):
        columns = rows[0][1]["media_columns"]
        table = parquet.read_row_group(number, columns=columns, use_threads=False)
        for index, row in rows:
            record = table.slice(index, 1).to_pylist()[0]
            if row["source"] == "TIGER-Lab/Mantis-Eval":
                inputs = record["images"]
            else:
                inputs = [record[column] for column in columns]
            media = [item["bytes"] for item in inputs if item and item.get("bytes")]
            for item, data in zip(row["images"], media, strict=True):
                path = media_file(item, assets, out)
                if not path.is_file():
                    save_media(item, data, path)


def restore(rows, assets, cache, out, allow_download):
    groups = defaultdict(list)
    for row in rows:
        if not all(media_file(item, assets, out).is_file() for item in row["images"]):
            groups[original_source(row)].append(row)
    downloads = out / "source_cache"
    problems = download_sources([v[0] for v in groups.values()], cache, downloads) if allow_download else {}
    for original, group in sorted(groups.items()):
        local, _ = source_location(group[0], original)
        file = cache / local if (cache / local).is_file() else downloads / local
        if not file.is_file():
            problems.setdefault(str(local), "Missing pinned source in available caches")
            continue
        try:
            function = restore_pdf if group[0]["document"] else restore_parquet
            function(group, file, assets, out)
            print(f"MEDIA_READY {local} {len(group)} requests", flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            problems[str(local)] = f"{type(error).__name__}: {error}"
    return problems


def relocated(rows, assets, out):
    exports, missing = [], []
    for row in rows:
        tensor = assets / Path(row["token_metadata_file"]).relative_to(ORIGINAL_ASSETS)
        files = [media_file(item, assets, out) for item in row["images"]]
        if not tensor.is_file() or not all(file.is_file() for file in files):
            missing.append(row["request_id"])
            continue
        if file_digest(tensor) != row["token_metadata_SHA256"]:
            raise ValueError(f"Unknown metadata bytes in {row['request_id']}")
        images = []
        for item, path in zip(row["images"], files, strict=True):
            if file_digest(path) != item["SHA256"]:
                raise ValueError(f"Unknown image bytes in {row['request_id']}")
            images.append({**item, "path":str(path)})
        exports.append({**row, "images":images, "token_metadata_file":str(tensor)})
    return exports, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    assets, cache, out = (path.absolute() for path in (args.assets, args.cache, args.out))
    canonical = subprocess.check_output(["git", "-C", str(args.repository), "show", f"{CANONICAL}:{MAIN400}"])
    if digest(canonical) != MAIN400_SHA256:
        raise ValueError("Canonical Main400 SHA256 mismatch")
    rows = json.loads(canonical)["requests"]
    problems = restore(rows, assets, cache, out, args.allow_download)
    qualified, missing = relocated(rows, assets, out)
    filename = f"main400_media_n{len(qualified)}.json"
    sha256 = write(out / filename, {"name":filename[:-5], "actual_count":len(qualified), "requests":qualified})
    write(out / "main400_media_receipt.json", {"canonical_SHA256":MAIN400_SHA256, "canonical_requests":len(rows),
                                              "filename":filename, "SHA256":sha256,
                                              "qualified_requests":len(qualified),
                                              "missing_request_ids":missing, "source_problems":problems})
    print(f"Media qualified {len(qualified)}/{len(rows)} requests at {filename}", flush=True)


if __name__ == "__main__":
    main()
