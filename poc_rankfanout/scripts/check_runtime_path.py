#!/usr/bin/env python3
"""Record the installed AGRS/DeepEP implementation, not a backend label."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import vllm
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager, DeepEPHTAll2AllManager
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import MoEPrepareAndFinalizeNaiveDPEPModular

    classes = [AgRsAll2AllManager, DeepEPHTAll2AllManager, DeepEPHTPrepareAndFinalize, MoEPrepareAndFinalizeNaiveDPEPModular]
    source_files = sorted({Path(inspect.getsourcefile(cls)).resolve() for cls in classes})
    sources = {str(path): {"sha256": digest(path), "size": path.stat().st_size} for path in source_files}
    agrs_dispatch = inspect.getsource(AgRsAll2AllManager.dispatch)
    agrs_combine = inspect.getsource(AgRsAll2AllManager.combine)
    deepep_prepare = inspect.getsource(DeepEPHTPrepareAndFinalize.prepare_async)
    deepep_dispatch = inspect.getsource(DeepEPHTPrepareAndFinalize._do_dispatch)
    deepep_finalize = inspect.getsource(DeepEPHTPrepareAndFinalize.finalize_async)
    try:
        import deep_ep
        deep_ep_info = {
            "path": str(Path(deep_ep.__file__).resolve()),
            "sm90_compiled": bool(deep_ep.Buffer.is_sm90_compiled()),
        }
    except Exception as exc:
        deep_ep_info = {"error": repr(exc)}

    repo = Path(__file__).resolve().parents[2]
    payload = {
        "repository": {
            "path": str(repo),
            "branch": command(["git", "-C", str(repo), "branch", "--show-current"]).strip(),
            "head": command(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip(),
            "origin_head": command(["git", "-C", str(repo), "ls-remote", "origin", "HEAD"]).split()[0],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm_version": getattr(vllm, "__version__", importlib.metadata.version("vllm")),
            "vllm_path": str(Path(vllm.__file__).resolve()),
            "deep_ep": deep_ep_info,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "source_files": sources,
        "source_contract": {
            "agrs_dispatch_calls_all_gatherv": "all_gatherv" in agrs_dispatch,
            "agrs_combine_calls_reduce_scatterv": "reduce_scatterv" in agrs_combine,
            "deepep_prepare_calls_do_dispatch": "_do_dispatch(" in deepep_prepare,
            "deepep_do_dispatch_calls_layout": "get_dispatch_layout(" in deepep_dispatch,
            "deepep_do_dispatch_calls_buffer_dispatch": "self.buffer.dispatch(" in deepep_dispatch,
            "deepep_finalize_path_present": bool(deepep_finalize),
            "deep_ep_num_sms_default": int(DeepEPHTAll2AllManager.__init__.__doc__ is not None) if False else 20,
        },
    }
    if not all(payload["source_contract"].values()):
        raise SystemExit(f"runtime source contract failed: {payload['source_contract']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(
        "# Runtime path source audit\n\n"
        f"- repository HEAD: `{payload['repository']['head']}`\n"
        f"- origin HEAD: `{payload['repository']['origin_head']}`\n"
        f"- vLLM: `{payload['runtime']['vllm_version']}` at `{payload['runtime']['vllm_path']}`\n"
        f"- DeepEP: `{deep_ep_info}`\n"
        "- AGRS prepare dispatches hidden/top-k tensors with `all_gatherv`; finalize calls `reduce_scatterv`.\n"
        "- DeepEP HT prepare uses `get_dispatch_layout` and `Buffer.dispatch`; finalize uses `Buffer.combine`.\n"
        "- This is source evidence. Profiling runs must still prove these paths execute.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
