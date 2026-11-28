"""Run one isolated Native A/A and shared-encoder Native/D-local logit check."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.capacity import PULL_PARAMS, headers
from ..mllm_specific_disaggregation_discovery.native_epd import prompt_messages


PORTS = {"encoder": 18840, "prefill": 18841, "decode": 18842}
COMPLETION = "v1/chat/completions"


def call(port, path, body=None, request_headers=None):
    request = Request(f"http://127.0.0.1:{port}/{path}",
                      data=None if body is None else json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", **(request_headers or {})})
    with urlopen(request, timeout=480) as response:
        return json.load(response)


def verify_workers(run):
    for role, port in PORTS.items():
        saved = rt.read(run / "state" / f"worker_{role}.json")
        owner = rt.read(run / "state" / f"owned_{role}.json")
        live, = call(port, "campaign/worker_info")
        rt.require(live["pid"] == saved["pid"] and live["role"] == role
                   and rt.pid_owned(live["pid"], owner["sid"], owner["nonce"]),
                   f"Active worker for {role} differs from the owned comparison run")
        if role == "decode":
            environment = Path("/proc", str(live["pid"]), "environ").read_bytes().split(b"\0")
            rt.require(b"MLLM_EPD_LATE_CORRECTNESS=1" in environment,
                       "Launch the owned D worker with MLLM_EPD_LATE_CORRECTNESS=1 for the existing logit probe")


def state():
    value, = call(PORTS["encoder"], "campaign/ec_ipc/state")
    return value


@contextmanager
def shared_encoding(body, base, row):
    encoder = PORTS["encoder"]
    initial = state()
    rt.require(initial["live_images"] == initial["outstanding_exports"] == 0, "Encoder must be quiescent")
    inline, = call(encoder, "campaign/ec_ipc/inline_policy", {"enabled": False})
    previous_role = inline["before"]
    metadata_key = headers(base, "E")["x-campaign-metadata-key"]
    try:
        result = call(encoder, COMPLETION, dict(body, max_tokens=1), headers(base, "E:full"))
        rt.require(result["usage"]["prompt_tokens"] == row["total_input_tokens"], "Original encoder prompt changed")
        held = state()
        rt.require(held["live_images"] == row["image_count"]
                   and held["saved_count"] - initial["saved_count"] == row["image_count"],
                   "Expected one new original encoder export per request image")
        yield metadata_key, held
    finally:
        current = state()
        if current["live_images"]:
            current, = call(encoder, "campaign/ec_ipc/release", {"key": metadata_key})
        rt.require(current["live_images"] == current["outstanding_exports"] == 0,
                   "Shared encoder ownership did not drain")
        restore = {"enabled": previous_role is not None}
        if previous_role is not None:
            restore["consumer_role"] = previous_role
        restored, = call(encoder, "campaign/ec_ipc/inline_policy", restore)
        rt.require(restored["inline_role"] == previous_role, "Encoder IPC policy was not restored")


def stage_headers(base, stage, metadata_key):
    return dict(headers(base, stage), **{"x-campaign-metadata-key": metadata_key})


def complete(body, base, metadata_key, *, native, prompt_count):
    request = dict(body)
    if native:
        prefill = call(PORTS["prefill"], COMPLETION, dict(request, max_tokens=1,
                       kv_transfer_params=dict(PULL_PARAMS)), stage_headers(base, "P", metadata_key))
        handoff = prefill["kv_transfer_params"]
        rt.require(handoff["do_remote_prefill"] and handoff["remote_block_ids"]
                   and handoff["remote_num_tokens"] == prompt_count, "Incomplete native NIXL prompt export")
        request["kv_transfer_params"] = handoff
    result = call(PORTS["decode"], COMPLETION, request, stage_headers(base, "D", metadata_key))
    ids = result["choices"][0]["token_ids"]
    rt.require(len(ids) == body["max_tokens"] and result["usage"]["completion_tokens"] == len(ids)
               and result["usage"]["prompt_tokens"] == prompt_count, "Expected full original fixed-length decode")
    prompt = result["prompt_token_ids"]
    rt.require(prompt and len(prompt) == prompt_count, "Original full prompt token IDs are unavailable")
    return {"id": base, "token_ids": ids, "prompt_token_ids": prompt,
            "route": "E_P_D" if native else "E_D", "metadata_key": metadata_key}


@contextmanager
def probe(run, name, ids, targets):
    port = PORTS["decode"]
    receipt, = call(port, "campaign/greedy_probe/begin", {"name": name, "ids": ids, "targets": targets})
    expected = run / "raw" / "greedy_probe" / name
    try:
        rt.require(Path(receipt["folder"]).resolve() == expected.resolve(),
                   "Greedy probe belongs to another worker session")
        yield expected / "manifest.json"
    finally:
        call(port, "campaign/greedy_probe/end", {})


def tensor_metrics(left, right):
    import torch

    rt.require(left.shape == right.shape and left.dtype == right.dtype, "Full logit vocab layouts differ")
    rt.require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), "Non-finite value in captured logits")
    delta = right.float() - left.float()
    return {
        "vocab_size": left.numel(), "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs": float(delta.abs().max().item()), "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float((delta.norm() / left.float().norm().clamp_min(1e-12)).item()),
    }


def capture_geometry(manifest, sample, base, step):
    event = manifest["events"][sample["event"]]
    request, = event["requests"]
    rt.require(request["base"] == base and request["step"] == step, "Probe selected a different request or step")
    return {**{field: request[field] for field in ("start", "query_tokens", "prompt_tokens")},
            "padded_query_rows": event["padded_query_rows"], "graph_mode": event["graph_mode"]}


def compare(names, paths, outputs, teacher):
    from contextlib import ExitStack
    from safetensors import safe_open

    first, native, local = names
    steps = len(teacher) + 1
    values = [rt.read(path) for path in paths]
    manifests = [values[0], values[1], values[1]]
    rt.require(not manifests[0]["teacher_forced"] and manifests[1]["teacher_forced"], "Probe mode changed")
    rt.require(outputs[0]["prompt_token_ids"] == outputs[1]["prompt_token_ids"] == outputs[2]["prompt_token_ids"],
               "The original prompt token identities changed")
    rt.require(all(result["token_ids"][:steps - 1] == teacher for result in outputs),
               "The forced arms did not emit the shared teacher prefix")
    comparisons = []
    with ExitStack() as stack:
        tensors = []
        for manifest in values:
            receipt = manifest["receipt"]
            field = Path(receipt["path"])
            rt.require(rt.sha(field) == receipt["sha256"], "Greedy probe tensor digest mismatch")
            tensors.append(stack.enter_context(safe_open(field, framework="pt", device="cpu")))
        for step in range(steps):
            keys = [f"{name}.{step}" for name in names]
            samples = [manifest["samples"][key] for manifest, key in zip(manifests, keys, strict=True)]
            expected_input = (outputs[0]["prompt_token_ids"][-1] if step == 0 else teacher[step - 1])
            rt.require(all(sample["input_token_id"] == expected_input for sample in samples),
                       f"Actual input token differs at step {step}")
            rt.require(all(sample["mrope_position_ids"] == samples[0]["mrope_position_ids"] for sample in samples),
                       f"Actual MRoPE differs at step {step}")
            if step < steps - 1:
                rt.require(all(sample["forced_id"] == teacher[step] for sample in samples[1:]),
                           f"Teacher forcing differs at step {step}")
            a0, a1, b = [archive.get_tensor(key + ".logits")
                         for archive, key in zip([tensors[0], tensors[1], tensors[1]], keys, strict=True)]
            comparisons.append({
                "step": step, "actual_input_token": expected_input,
                "mrope_position_ids": samples[0]["mrope_position_ids"],
                "native_AA": tensor_metrics(a0, a1), "native_vs_D_local": tensor_metrics(a1, b),
                "greedy_ids_before_forcing": {name: sample["greedy_id"]
                                              for name, sample in zip(names, samples, strict=True)},
                "top1_top2_margins": [sample["top1_top2_margin"] for sample in samples],
                "geometry": [capture_geometry(manifest, sample, name, step) for manifest, sample, name in
                             zip(manifests, samples, names, strict=True)],
            })
    counts = {
        "all_steps": steps,
        "native_AA_greedy_agreement": sum(row["greedy_ids_before_forcing"][first] ==
                                          row["greedy_ids_before_forcing"][native] for row in comparisons),
        "native_vs_D_greedy_agreement": sum(row["greedy_ids_before_forcing"][native] ==
                                            row["greedy_ids_before_forcing"][local] for row in comparisons),
    }
    return counts, comparisons


def run_pair(root, manifest, index, tag, output_steps=16):
    """Run in an idle owned E/P/D lease and return the compact result path."""
    rt.require(index >= 0 and re.fullmatch(r"[A-Za-z0-9_-]+", tag), "Invalid probe name or index")
    rt.require(output_steps in (16, 32), "Use a supported output length")
    root = root.resolve(strict=True)
    output = root / "analysis" / f"{tag}_logit_pair.json"
    rt.require(not output.exists(), "Use a new correctness tag")
    row = rt.read(manifest)["requests"][index]
    rt.require(row["images"] and row["total_input_tokens"] + output_steps <= 32768,
               "Invalid image prompt or context length")
    verify_workers(root)
    names = tuple(f"{tag}_{name}" for name in ("native_unforced", "native_teacher", "d_teacher"))
    body = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "messages": prompt_messages(row, tag),
        "temperature": 0, "top_p": 1, "seed": 2026112401, "max_tokens": output_steps,
        "ignore_eos": True, "return_token_ids": True, "stream": False,
    }
    with shared_encoding(body, tag, row) as (metadata_key, encoder_receipt):
        with probe(root, tag + "_baseline", [names[0]], {names[0]: {"steps": output_steps}}) as control:
            baseline = complete(body, names[0], metadata_key, native=True, prompt_count=row["total_input_tokens"])
        teacher = baseline["token_ids"][:-1]
        targets = {name: {"steps": output_steps, "prefix": teacher} for name in names[1:]}
        with probe(root, tag + "_forced", list(names[1:]), targets) as paired:
            native = complete(body, names[1], metadata_key, native=True, prompt_count=row["total_input_tokens"])
            local = complete(body, names[2], metadata_key, native=False, prompt_count=row["total_input_tokens"])
    counts, steps = compare(names, (control, paired), (baseline, native, local), teacher)
    rt.write(output, {
        "evidence": "CORRECTNESS_ONLY", "request_id": row["request_id"],
        "prompt_sha256": row["prompt_SHA256"], "manifest_sha256": rt.sha(manifest),
        "shared_E_receipt": encoder_receipt, "teacher_prefix_sha256": sha256(json.dumps(teacher).encode()).hexdigest(),
        "probe_manifests": [{"path": str(path), "sha256": rt.sha(path)} for path in (control, paired)],
        "counts": counts, "steps": steps,
        "scope": "Three serialized D runs on one held actual E output. All logits are original full-vocab "
                 "values before teacher-forced token selection. Diagnostics copy values to CPU and cannot "
                 "be used for timing. Same-prefix numerical and greedy agreement describes this request.",
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, choices=(16, 32), default=16)
    args = parser.parse_args()
    path = run_pair(args.run_root, args.manifest, args.index, args.tag, args.steps)
    print(json.dumps(rt.read(path)["counts"]))


if __name__ == "__main__":
    main()
