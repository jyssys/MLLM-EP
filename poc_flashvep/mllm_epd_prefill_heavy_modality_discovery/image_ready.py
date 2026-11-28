"""Observe the native encoder transport events without changing their ownership."""

import atexit
from collections import deque
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import threading
import time


POLL_SECONDS = .005
FLUSH_SECONDS = .25
EVENT_TIMEOUT_NS = 15_000_000_000
MAX_INFLIGHT = 4096
MAX_QUERIES_PER_PASS = 256
MAX_MANIFESTS = 16384
REQUEST_KEY = re.compile(r"[a-f0-9]{64}")


class ImageReadyObserver:
    def __init__(self, registry, metadata_root, output_dir):
        self.registry = registry
        self.original_save = registry.save
        self.metadata_root = metadata_root
        self.output_dir = output_dir
        self.pid = os.getpid()
        self.before_install = registry.saved_count
        self.installed_ns = time.monotonic_ns()
        self.events_file = output_dir / f"events_{self.pid}.jsonl"
        self.manifests_file = output_dir / f"manifests_{self.pid}.jsonl"
        self.receipt_file = output_dir / f"receipt_{self.pid}.json"
        self.incoming = SimpleQueue()
        self.slots = threading.BoundedSemaphore(MAX_INFLIGHT)
        self.stop_event = threading.Event()
        self.pending = deque()
        self.seen_manifests = set()
        self.enqueued = self.dropped = self.completed = self.unresolved = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, name="native-image-ready-poll", daemon=True)

    def save(self, key, tensor):
        begin = time.monotonic_ns()
        result = self.original_save(key, tensor)
        returned = time.monotonic_ns()
        if self.stop_event.is_set():
            return result
        ready = self.registry.ready.get(key)
        if ready is None or not self.slots.acquire(blocking=False):
            self.dropped += 1
            return result
        self.incoming.put((ready, {
            "producer_pid": self.pid,
            "image_identifier": key,
            "embedding_rows": int(tensor.shape[0]),
            "save_begin_ns": begin,
            "save_returned_ns": returned,
            "completion_lower_bound_ns": begin,
            "completion_upper_bound_ns": None,
            "query_count": 0,
            "evidence": "DIAGNOSTIC",
        }))
        self.enqueued += 1
        return result

    def poll(self, completed_rows):
        for _ in range(min(len(self.pending), MAX_QUERIES_PER_PASS)):
            event, row = self.pending.popleft()
            begin = time.monotonic_ns()
            row["query_count"] += 1
            if begin - row["save_returned_ns"] > EVENT_TIMEOUT_NS:
                row["status"] = "QUERY_TIMEOUT"
            else:
                try:
                    complete = event.query()
                except RuntimeError as exc:
                    row.update(status="QUERY_ERROR", error=str(exc))
                else:
                    if not complete:
                        row["completion_lower_bound_ns"] = begin
                        self.pending.append((event, row))
                        continue
                    row.update(status="CUDA_EVENT_COMPLETE",
                               completion_upper_bound_ns=time.monotonic_ns())
            completed_rows.append(row)
            self.slots.release()
            if row["status"] == "CUDA_EVENT_COMPLETE":
                self.completed += 1
            else:
                self.unresolved += 1

    def new_manifests(self):
        for folder in self.metadata_root.iterdir():
            if len(self.seen_manifests) >= MAX_MANIFESTS:
                break
            key = folder.name
            if key in self.seen_manifests or not REQUEST_KEY.fullmatch(key) or folder.is_symlink():
                continue
            path = folder / "manifest.json"
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue
            value = json.loads(raw)
            images = value["image_hashes"]
            if value["key"] != key or len(images) != value["images"]:
                raise ValueError(f"Invalid processor manifest for {key}")
            self.seen_manifests.add(key)
            yield {"request_key": key, "ordered_image_identifiers": images,
                   "image_spans": value.get("image_spans"), "lm_input_tokens": value["prompt_tokens"],
                   "actual_manifest_sha256": sha256(raw).hexdigest()}

    @staticmethod
    def append_rows(path, rows):
        if rows:
            with path.open("a", encoding="utf-8") as stream:
                stream.writelines(json.dumps(row) + "\n" for row in rows)

    def receipt(self):
        return {
            "enabled": True, "evidence": "DIAGNOSTIC", "producer_pid": self.pid,
            "installed_ns": self.installed_ns, "native_saves_before_install": self.before_install,
            "enqueued": self.enqueued, "dropped": self.dropped, "complete": self.completed,
            "unresolved": self.unresolved, "manifest_count": len(self.seen_manifests),
            "manifest_limit_reached": len(self.seen_manifests) >= MAX_MANIFESTS,
            "event_rows": str(self.events_file), "request_mapping": str(self.manifests_file),
            "error": self.error,
            "join": "SHA-256 of the client runtime_request_id gives request_key. "
                    "Join image_identifier against the actual ordered processor manifest.",
            "scope": "One existing producer-stream event per original EC save. The successful query bounds its "
                     "completion in the host's monotonic clock. Native save publishes the host marker before "
                     "returning. Its publication and GPU event completion can occur in either order. "
                     "The native runner saves all items after submitting all encoder groups in the pass. "
                     "Events are transport-readiness witnesses, without individual vision-kernel timing, "
                     "consumer clone completion, or an earlier route opportunity.",
        }

    def run(self):
        rows = []
        flush_due = time.monotonic() + FLUSH_SECONDS
        try:
            while not self.stop_event.is_set():
                try:
                    record = self.incoming.get(timeout=0 if self.pending else FLUSH_SECONDS)
                    if record is not None:
                        self.pending.append(record)
                    while not self.incoming.empty():
                        record = self.incoming.get_nowait()
                        if record is not None:
                            self.pending.append(record)
                except Empty:
                    pass
                self.poll(rows)
                if len(rows) >= MAX_QUERIES_PER_PASS or time.monotonic() >= flush_due:
                    self.append_rows(self.events_file, rows)
                    rows.clear()
                    if self.enqueued:
                        self.append_rows(self.manifests_file, list(self.new_manifests()))
                    flush_due = time.monotonic() + FLUSH_SECONDS
                if self.pending:
                    self.stop_event.wait(POLL_SECONDS)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()
        finally:
            while not self.incoming.empty():
                record = self.incoming.get_nowait()
                if record is not None:
                    self.pending.append(record)
            while self.pending:
                _, row = self.pending.popleft()
                row["status"] = "OBSERVER_STOPPED_WITHOUT_COMPLETION"
                rows.append(row)
                self.unresolved += 1
                self.slots.release()
            try:
                self.append_rows(self.events_file, rows)
                if self.enqueued:
                    self.append_rows(self.manifests_file, list(self.new_manifests()))
            finally:
                self.receipt_file.write_text(json.dumps(self.receipt(), indent=2) + "\n", encoding="utf-8")

    def close(self):
        self.stop_event.set()
        self.incoming.put(None)
        self.thread.join(timeout=1)


def install_image_ready_observer():
    """Enable once before warmup via the E worker's readiness RPC."""
    if os.environ.get("MLLM_EPD_IMAGE_READY") != "1" or os.environ.get("MLLM_EPD_ROLE") != "encoder":
        return {"enabled": False}
    from ..mllm_epd_prefill_heavy_token_split.common import ROOT
    from ..mllm_epd_prefill_heavy_token_split.cuda_ipc_ec import REGISTRY

    if REGISTRY is None:
        return {"enabled": False, "reason": "Native encoder CUDA IPC producer is unavailable"}
    observer = getattr(REGISTRY, "_image_ready_observer", None)
    if observer is None:
        output = ROOT / "raw" / "image_ready"
        output.mkdir(exist_ok=True)
        observer = ImageReadyObserver(REGISTRY, Path(os.environ["MLLM_EPD_METADATA_PATH"]), output)
        REGISTRY._image_ready_observer = observer
        REGISTRY.save = observer.save
        observer.thread.start()
        atexit.register(observer.close)
    return observer.receipt()
