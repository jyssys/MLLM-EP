"""Record installed metadata without importing CUDA-capable research packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            timeout=30, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    return {"returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    versions = """
import importlib.metadata as m, importlib.util, json, sys
names = ['torch', 'triton', 'transformers', 'vllm', 'sglang', 'flash-attn',
         'flashinfer-python', 'deep-ep', 'nanoflow', 'gurobipy', 'nvmath-python']
result = {'python': sys.version, 'executable': sys.executable, 'packages': {}}
for name in names:
    try: result['packages'][name] = m.version(name)
    except m.PackageNotFoundError: result['packages'][name] = None
result['top_level_module_locations'] = {}
for name in ['flash_attn', 'flash_attn_2_cuda', 'nanovllm', 'nanoflow', 'deep_ep', 'vllm']:
    spec = importlib.util.find_spec(name)
    result['top_level_module_locations'][name] = spec.origin if spec is not None else None
result['metadata_caveat'] = 'A null distribution version does not imply a source/native module is unavailable.'
print(json.dumps(result))
"""
    environments = {}
    for name in ("scheduling-fastpp-py310", "scheduling-layered-py310",
                 "scheduling-nanoflow-py310", "flashvep-deepep-v020"):
        result = command([f"/home/esjung/.venvs/{name}/bin/python", "-c", versions])
        environments[name] = json.loads(result["stdout"]) if result["returncode"] == 0 else result
    references = {}
    for name in ("FastPP", "layered-prefill", "Nanoflow-h100"):
        path = args.results / "refs" / name
        diff = command(["git", "diff", "--binary"], path)
        references[name] = {
            "commit": command(["git", "rev-parse", "HEAD"], path)["stdout"].strip(),
            "tracked_status": command(["git", "status", "--short", "--untracked-files=no"], path)["stdout"],
            "tracked_diff_sha256": hashlib.sha256(diff["stdout"].encode()).hexdigest(),
            "diff_command_returncode": diff["returncode"],
        }
    patches = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in root.rglob("*.patch")}
    result = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu_scope": [4, 5, 6, 7],
        "metadata_collection_cuda_visible_devices": "",
        "no_cuda_capable_package_imports": True,
        "environments": environments, "official_references": references,
        "patch_sha256": patches,
        "authorized_gpu_inventory": command([
            "nvidia-smi", "-i", "4,5,6,7", "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader"]),
        "note": "NVML inventory is read-only, not a GPU experiment; binary builds stay local",
    }
    target = args.results / "cpu_analysis" / "environment_provenance.json"
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({"artifact": str(target), "environments": len(environments),
                      "official_references": len(references), "patches": len(patches)}))


if __name__ == "__main__":
    main()
