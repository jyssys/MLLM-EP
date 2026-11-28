"""Export a compact, reviewable CPU-side evidence snapshot for the EPD reducer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .nixl_intervals import handoff_key


SCHEMA = 4
CHUNK = 1024 * 1024
SECTIONS = ("configs", "raw", "analysis", "state", "logs")
PROFILE_MEMBERS = {
    "actual_ROI.json", "kernels.csv", "kernels.csv.names.csv", "provenance.json",
    "scheduler.json", "worker_calls.json", "copies.csv", "copy.csv", "memcpy.csv",
}
STATE_MEMBERS = {"complete.json", "environment.json", "model_download_complete.json", "failure.json"}
KV_TRANSFER_FIELDS = {
    "do_remote_prefill", "do_remote_decode", "remote_num_tokens", "tp_size", "remote_block_count",
    "remote_block_id_counts", "remote_handoff_SHA256",
}
ROSTER_FIELDS = {
    "request_id", "prompt_SHA256", "total_input_tokens", "visual_tokens", "text_tokens",
    "image_count", "source", "stratum", "grid_thw", "structural_tokens",
}
REQUEST_FIELDS = {
    "request_id", "runtime_request_id", "index", "scheduled_arrival", "dispatch",
    "dispatch_lateness", "payload_SHA256", "prompt_SHA256", "input_tokens_manifest",
    "input_tokens", "source", "stratum", "image_count", "status", "token_ids",
    "token_times", "usage", "HTTP_status", "finish_reason", "finish", "output_tokens",
    "TTFT_ms", "ITL_ms", "TPOT_ms", "E2E_ms", "output_cap_hit", "submit",
    "visual_tokens", "visual_ratio", "replica", "error_type", "prompt_token_ids_SHA256",
}
REMOVED_FIELDS = {
    "text", "chunks", "choices", "prompt", "prompt_text", "prompt_tokens_ids",
    "prompt_ids", "prompt_token_ids", "prompt_logprobs", "messages", "question",
    "request_body", "response_body", "payload", "b64_json", "image_url",
    "image_data", "pixel_values", "embeddings", "visual_embeddings", "image_embeds",
    "ec_transfer_params", "remote_block_ids", "ipc_handle", "cuda_ipc_handle",
    "authorization", "headers", "cookie", "cookies", "token", "access_token",
    "api_key", "hf_token", "password", "credential", "credentials", "secret",
    "command", "cmdline", "argv", "environ", "env", "stderr", "stdout", "errors",
    "error", "error_body", "traceback", "remote_engine_id", "remote_request_id",
    "remote_host", "remote_port", "remote_blocks_expiry_time",
}


@dataclass(frozen=True)
class Source:
    path: Path
    name: str
    mode: str
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path, name, mode):
        state = path.stat()
        return cls(path, name, mode, state.st_size, state.st_mtime_ns)

    def stamp(self):
        return {"path": self.name, "bytes": self.size, "mtime_ns": self.mtime_ns, "mode": self.mode}


class Snapshot:
    """Read and fingerprint precisely the prefix visible when the source was inventoried."""

    def __init__(self, source, stream):
        self.stream = stream
        self.remaining = source.size
        self.sha = hashlib.sha256()

    def read(self, size):
        data = self.stream.read(min(self.remaining, size))
        self.sha.update(data)
        self.remaining -= len(data)
        return data

    def lines(self):
        while self.remaining:
            line = self.stream.readline(self.remaining)
            if not line:
                break
            self.sha.update(line)
            self.remaining -= len(line)
            yield line

    def drain(self):
        while self.remaining and self.read(CHUNK):
            pass


def capture_mode(relative):
    section, name = relative.parts[0], relative.name
    if section == "configs":
        if name == "frozen_plan.json":
            return "plan"
        if name == "cohorts.jsonl":
            return "jsonl"
        if name.endswith(".json"):
            return "json"
    if section == "analysis" and name.endswith(".json"):
        return "json"
    if section == "state" and (
        name in STATE_MEMBERS or name.endswith(("_complete.json", "_ownership.json"))
    ):
        return "json"
    if section == "raw":
        if re.fullmatch(r"nixl_read_decode_\d+\.jsonl", name):
            return "jsonl"
        if re.fullmatch(r"nixl_read_decode_\d+\.json", name):
            return "json"
        if relative.parts[1] == "profiles":
            if (name in PROFILE_MEMBERS and name.endswith(".json")) or re.fullmatch(r"worker-\d+\.json", name):
                return "json"
            basename = name.removesuffix(".gz")
            if basename in PROFILE_MEMBERS or re.fullmatch(r"worker-\d+\.csv", basename):
                return "verbatim"
        if relative.parts[1] == "image_ready":
            if re.fullmatch(r"(?:events|manifests)_\d+\.jsonl", name):
                return "jsonl"
            if re.fullmatch(r"receipt_\d+\.json", name):
                return "json"
        if name.endswith("_prometheus.json"):
            return "verbatim_json"
        if name.endswith(("_gpm.json", "_rows.json", "_prepared.json", "_metrics.json",
                          "_topology.json", "_probe.json", "_profile_receipts.json")):
            return "json"
        if name.endswith(("_requests.jsonl", "_completion_journal.jsonl")):
            return "client"
        if name.startswith("proxy_") and name.endswith(".jsonl"):
            return "proxy"
        if name.endswith("_scheduler.jsonl") or (
            name.startswith(("processor_handoff_", "ec_cuda_ipc_", "ec_transfer_"))
            and name.endswith(".jsonl")
        ):
            return "jsonl"
    return "digest-only"


def encode(data):
    return (json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def block_count(value):
    if isinstance(value, list):
        return sum(block_count(item) for item in value)
    return int(isinstance(value, int) and not isinstance(value, bool))


def client_fields(data):
    record = {key: value for key, value in data.items() if key in REQUEST_FIELDS}
    token_ids = data.get("prompt_token_ids")
    if isinstance(token_ids, list):
        canonical = json.dumps(token_ids, separators=(",", ":")).encode()
        record["prompt_token_ids_SHA256"] = hashlib.sha256(canonical).hexdigest()
    return record


def clean(data):
    if isinstance(data, list):
        return [clean(item) for item in data]
    if not isinstance(data, dict):
        return data
    if "prompt_token_ids" in data and ("chunks" in data or "scheduled_arrival" in data):
        return clean(client_fields(data))
    answer = {}
    if "post_begin_ns" in data and (handoff := handoff_key(data)):
        answer["remote_handoff_SHA256"] = handoff
    for key, value in data.items():
        if key == "kv_transfer_params" and isinstance(value, dict):
            params = {field: value[field] for field in KV_TRANSFER_FIELDS if field in value}
            if handoff := handoff_key(value):
                params["remote_handoff_SHA256"] = handoff
            ids = value.get("remote_block_ids")
            if ids is not None:
                params["remote_block_count"] = block_count(ids)
                if isinstance(ids, list) and any(isinstance(item, list) for item in ids):
                    params["remote_block_id_counts"] = [block_count(item) for item in ids]
            answer[key] = params
        elif key.lower() not in REMOVED_FIELDS and not key.lower().endswith(("_secret", "_password", "_api_key")):
            answer[key] = clean(value)
    return answer


def sanitize_line(data, mode):
    if mode == "client":
        if "request_id" in data:
            data = client_fields(data)
        elif isinstance(data.get("row"), dict):
            data = {key: value for key, value in data.items() if key in {"row", "event", "time"}}
            data["row"] = sanitize_line(data["row"], mode)
    return clean(data)


def collect_sources(epoch):
    sources = []
    for section in SECTIONS:
        for path in sorted((epoch / section).rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(epoch)
            sources.append(Source.capture(path, str(relative), capture_mode(relative)))
    return sources


def collect_rosters(epoch, sources):
    plan = next((source for source in sources if source.mode == "plan"), None)
    if plan is None:
        return [], {}
    try:
        with plan.path.open("rb") as stream:
            cases = json.loads(stream.read(plan.size))["cases"]
    except (ValueError, KeyError):
        return [], {}
    rosters, references = [], {}
    approved = epoch.parent.resolve()
    for case in cases:
        original = case.get("manifest")
        if not original or original in references:
            continue
        path = Path(original)
        if not path.is_absolute():
            path = approved / path
        path = path.resolve()
        if not path.is_relative_to(approved) or not path.is_file():
            continue
        source = Source.capture(path, str(path.relative_to(approved)), "roster")
        with source.path.open("rb") as stream:
            snapshot = Snapshot(source, stream)
            snapshot.drain()
        if snapshot.remaining:
            raise OSError(f"Roster changed during capture: {source.name}")
        alias = f"dataset/{snapshot.sha.hexdigest()[:16]}__{path.name}"
        if alias not in references.values():
            rosters.append(source)
        references[original] = alias
        references[str(path)] = alias
    return rosters, references


def add_member(tar, fileobj, name, size):
    fileobj.seek(0)
    digest = hashlib.file_digest(fileobj, "sha256").hexdigest()
    fileobj.seek(0)
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o644
    tar.addfile(member, fileobj)
    return {"archive_name": name, "archive_bytes": size, "archive_sha256": digest}


def write_source(tar, source, epoch, references):
    if source.mode == "roster":
        name = references[str(source.path)]
    else:
        name = f"{epoch.name}/{source.name}"
    receipt = {**source.stamp(), "status": source.mode}
    with source.path.open("rb") as stream, tempfile.SpooledTemporaryFile(max_size=4 * CHUNK) as out:
        snapshot = Snapshot(source, stream)
        if source.mode in {"jsonl", "client", "proxy"}:
            incomplete = invalid = kept = 0
            for line in snapshot.lines():
                if not line.endswith(b"\n"):
                    incomplete += 1
                    continue
                if not line.strip():
                    continue
                try:
                    out.write(encode(sanitize_line(json.loads(line), source.mode)))
                    kept += 1
                except (ValueError, TypeError):
                    invalid += 1
            receipt.update(records=kept, incomplete_records=incomplete, invalid_records=invalid)
        elif source.mode in {"json", "plan", "roster", "verbatim_json"}:
            raw = snapshot.read(source.size)
            try:
                data = json.loads(raw)
                if source.mode == "verbatim_json":
                    out.write(raw)
                elif source.mode == "roster":
                    out.write(encode({"requests": [
                        {key: value for key, value in row.items() if key in ROSTER_FIELDS}
                        for row in data["requests"]
                    ]}))
                    suffix = hashlib.sha256(raw).hexdigest()[:16]
                    if not Path(name).name.startswith(suffix + "__"):
                        raise OSError(f"Roster changed after reference capture: {source.name}")
                else:
                    if source.mode == "plan":
                        for point in data["cases"]:
                            if point.get("manifest"):
                                original = point["manifest"]
                                point["manifest"] = references.get(original, Path(original).name)
                    out.write(encode(clean(data)))
            except (ValueError, KeyError):
                receipt["status"] = "incomplete_or_invalid_json_digest_only"
        elif source.mode == "verbatim":
            while snapshot.remaining:
                chunk = snapshot.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        snapshot.drain()
        receipt.update(source_sha256=snapshot.sha.hexdigest(), source_bytes=source.size - snapshot.remaining)
        if snapshot.remaining:
            receipt["status"] = "short_read_digest_only"
        if "digest_only" not in receipt["status"] and out.tell():
            receipt.update(add_member(tar, out, name, out.tell()))
    return receipt


def export_epoch(epoch, out):
    sources = collect_sources(epoch)
    rosters, references = collect_rosters(epoch, sources)
    if not sources:
        return None
    archive = out / f"{epoch.name}.tar.gz"
    manifest = out / f"{epoch.name}.manifest.json"
    stamp = [source.stamp() for source in sources + rosters]
    if archive.is_file() and manifest.is_file():
        try:
            previous = json.loads(manifest.read_bytes())
        except ValueError:
            previous = {}
        if previous.get("schema") == SCHEMA and previous.get("snapshot") == stamp:
            if previous.get("archive_bytes") == archive.stat().st_size:
                with archive.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest == previous.get("archive_sha256"):
                    return {"epoch": epoch.name, "archive": archive.name, "status": "unchanged",
                            "bytes": previous["archive_bytes"], "sha256": digest}
    with tempfile.TemporaryDirectory(prefix=f".{epoch.name}-", dir=out) as scratch:
        staging = Path(scratch)
        pending_archive = staging / archive.name
        receipts = []
        with pending_archive.open("wb") as target:
            with gzip.GzipFile(fileobj=target, filename="", mode="wb", mtime=0, compresslevel=4) as packed:
                with tarfile.open(fileobj=packed, mode="w|") as tar:
                    for source in sources + rosters:
                        receipts.append(write_source(tar, source, epoch, references))
        with pending_archive.open("rb") as stream:
            archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        archive_bytes = pending_archive.stat().st_size
        pending_manifest = staging / manifest.name
        pending_manifest.write_text(json.dumps({
            "schema": SCHEMA,
            "epoch": epoch.name,
            "archive": archive.name,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "producer_marked_complete": any(source.name == "state/complete.json"
                                            for source in sources),
            "scope": list(SECTIONS),
            "snapshot": stamp,
            "sources": receipts,
        }, indent=2) + "\n")
        os.replace(pending_archive, archive)
        os.replace(pending_manifest, manifest)
    return {"epoch": epoch.name, "archive": archive.name, "status": "exported",
            "bytes": archive_bytes, "sha256": archive_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epoch", action="append", help="Limit export to one or more epoch directory names.")
    args = parser.parse_args()
    root = args.raw_root.resolve()
    if not root.is_dir():
        parser.error(f"Raw root is not a directory: {root}")
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.epoch or sorted({
        folder.name for folder in root.iterdir()
        if folder.is_dir() and (folder / "configs").is_dir() and (folder / "raw").is_dir()
    })
    for name in names:
        if Path(name).name != name or name in {".", ".."} or not (root / name).is_dir():
            parser.error(f"Unknown epoch directory: {name}")
        result = export_epoch(root / name, args.out)
        if result:
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
