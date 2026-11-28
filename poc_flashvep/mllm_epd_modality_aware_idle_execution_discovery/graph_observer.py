"""Build and use the D-side CUPTI CUDA graph envelope observer."""

import csv
import ctypes
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import time

from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import file_sha256, merged
from ..mllm_epd_prefill_heavy_modality_discovery.timeline import clock_pair


ARTIFACT = Path("references/graph_activity")
UNIT = "cuda_graph_launch_envelope"
SCOPE = ("CUDA graph execution start/end from CUPTI_GRAPH_TRACE. The timeline covers graph envelopes "
         "and includes spacing between graph nodes. Non-graph kernels, native connector activity, "
         "and inactivity of the device require separate observations.")


def load_build(run_dir):
    receipt = Path(run_dir) / ARTIFACT / "build.json"
    record = json.loads(receipt.read_text())
    if record["evidence_unit"] != UNIT or any(
        file_sha256(Path(record[key])) != record[f"{key}_SHA256"] for key in ("source", "library", "CUPTI_library")
    ):
        raise RuntimeError("Graph observer build receipt failed the hash or activity-unit check")
    return record


def build(run_dir, runtime):
    """Compile against the exact CUPTI and CUDA headers in the serving runtime."""
    target = Path(run_dir).resolve() / ARTIFACT
    active_source = Path(__file__).with_name("graph_activity.cc")
    receipt = target / "build.json"
    if receipt.is_file():
        record = load_build(run_dir)
        if record["source_SHA256"] != file_sha256(active_source):
            raise RuntimeError("Existing build uses another graph collector source revision")
        return record

    runtime = Path(runtime).resolve()
    candidates = sorted(runtime.glob("lib/python*/site-packages/nvidia/cuda_cupti"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one serving-runtime CUPTI installation under {runtime}")
    cupti, = candidates
    library_cupti = cupti / "lib/libcupti.so.12"
    compiler = runtime / "bin/x86_64-conda-linux-gnu-c++"
    cuda_include = runtime / "targets/x86_64-linux/include"
    if not all(path.exists() for path in (library_cupti, compiler, cuda_include, cupti / "include/cupti_activity.h")):
        raise RuntimeError("Serving runtime is missing its CUPTI library, CUDA headers or C++ compiler")
    target.mkdir(exist_ok=True, parents=True)
    source = target / active_source.name
    source.write_bytes(active_source.read_bytes())
    library = target / "graph_activity.so"
    command = [str(compiler), "-std=c++17", "-O2", "-fPIC", "-shared",
               "-I" + str(cuda_include), "-I" + str(cupti / "include"),
               str(source), str(library_cupti), "-Wl,-rpath," + str(library_cupti.parent), "-o", str(library)]
    subprocess.run(command, check=True)
    paths = {"source": source, "library": library, "CUPTI_library": library_cupti}
    record = {key: str(value) for key, value in paths.items()}
    record.update({f"{key}_SHA256": file_sha256(value) for key, value in paths.items()})
    record.update(command=command, evidence_unit=UNIT, scope=SCOPE)
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def read_envelopes(folder, left, right):
    """Align graph envelopes to host monotonic seconds without interpreting uncovered time."""
    folder = Path(folder)
    receipt = json.loads((folder / "actual_ROI.json").read_text())
    if receipt["evidence_unit"] != UNIT:
        raise ValueError(f"Unexpected activity unit in {folder}")
    counters = receipt["counters"]
    if (not receipt["passed"] or receipt["returncode"] or not counters["graph_records"]
            or any(counters[key] for key in ("invalid", "dropped", "unexpected", "overruns"))):
        raise ValueError(f"Incomplete graph observation in {folder}")
    coverage, (host, tick, scale), brackets = clock_pair(
        receipt["left"], receipt["right"], ("host_lower_ns", "host_upper_ns"))
    if not (coverage[0] <= left < right <= coverage[1] and .99 <= scale <= 1.01
            and 0 <= min(brackets) <= max(brackets) <= 1_000_000):
        raise ValueError(f"Graph clock does not cover the requested host ROI in {folder}")
    paths = [path for path in (folder / "graphs.csv", folder / "graphs.csv.gz") if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected one graph CSV in {folder}")
    path, = paths
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            start, end = int(row["start_ns"]), int(row["end_ns"])
            if start <= 0 or end <= start:
                raise ValueError(f"Invalid CUPTI graph envelope in {path}")
            records.append(((host + scale * (start - tick)) / 1e9, (host + scale * (end - tick)) / 1e9))
    if len(records) != counters["graph_records"]:
        raise ValueError(f"Graph record count disagrees with capture receipt in {folder}")
    return {"evidence_unit": UNIT, "scope": SCOPE, "graph_records": len(records),
            "graph_envelope_union": merged(records, left, right), "coverage": coverage,
            "clock_scale": scale, "max_anchor_bracket_ns": max(brackets)}


class GraphRecorder:
    """One graph-only capture per independent decode worker start."""

    def __init__(self, name, *, run_dir, role="decode", buffer_mib=16):
        if role != "decode" or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("Graph capture requires a valid profile name and the decode role")
        if type(buffer_mib) is not int or not 2 <= buffer_mib <= 128:
            raise ValueError("Graph capture requires 2–128 MiB of reserved host buffers")
        record = load_build(run_dir)
        self.folder = Path(run_dir).resolve() / "raw/profiles" / name / role
        self.folder.mkdir(parents=True)
        self.library = ctypes.CDLL(record["library"])
        signatures = {
            "prepare": [ctypes.c_size_t], "enable": [], "timestamp": [ctypes.POINTER(ctypes.c_uint64)],
            "close": [ctypes.c_char_p],
        }
        for key, args in signatures.items():
            entry = getattr(self.library, f"graph_activity_{key}")
            entry.argtypes, entry.restype = args, ctypes.c_int
        self.counters = ("records", "invalid", "dropped", "unexpected", "overruns",
                         "requested_buffers", "completed_buffers")
        for name in self.counters:
            counter = getattr(self.library, f"graph_activity_{name}")
            counter.argtypes, counter.restype = [], ctypes.c_uint64
        self._check("prepare", buffer_mib * 1024**2)
        self.left = None
        self.closed = False
        self._save("provenance.json", {"build": record, "role": role, "pid": os.getpid(), "buffer_mib": buffer_mib,
                                       "evidence_unit": UNIT, "scope": SCOPE})

    def _save(self, filename, value):
        with (self.folder / filename).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")

    def _check(self, operation, *args):
        status = getattr(self.library, f"graph_activity_{operation}")(*args)
        if status != 0:
            raise RuntimeError(f"CUPTI graph {operation} returned {status}")

    def _anchor(self):
        cupti = ctypes.c_uint64()
        before = time.monotonic_ns()
        self._check("timestamp", ctypes.byref(cupti))
        return {"cupti_ns": cupti.value, "host_lower_ns": before, "host_upper_ns": time.monotonic_ns()}

    def begin(self):
        import torch

        if self.left is not None:
            raise RuntimeError("Graph capture has previously started")
        torch.cuda.synchronize()
        self._check("enable")
        self.left = self._anchor()
        return {**self.left, "evidence_unit": UNIT}

    def close(self):
        import torch

        if self.left is None or self.closed:
            raise RuntimeError("Graph capture is inactive")
        torch.cuda.synchronize()
        right = self._anchor()
        code = self.library.graph_activity_close(os.fsencode(self.folder / "graphs.csv"))
        self.closed = True
        counters = {key: int(getattr(self.library, f"graph_activity_{key}")()) for key in self.counters}
        counters["graph_records"] = counters.pop("records")
        passed = (code == 0 and counters["graph_records"] > 0
                  and not any(counters[key] for key in ("invalid", "dropped", "unexpected", "overruns"))
                  and right["cupti_ns"] > self.left["cupti_ns"])
        result = {"evidence_unit": UNIT, "scope": SCOPE, "left": self.left, "right": right,
                  "counters": counters, "returncode": code, "passed": passed}
        self._save("actual_ROI.json", result)
        return result
