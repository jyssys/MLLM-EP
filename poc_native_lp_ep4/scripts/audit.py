import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from poc_native_lp_ep4.contract import MODEL,MODEL_REVISION,UPSTREAM_COMMIT,atomic_json


def cmd(*args):
    result=subprocess.run(args,text=True,capture_output=True)
    return {"args":args,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}


def main():
    import torch
    cfg=json.loads((Path(MODEL)/"config.json").read_text())
    text=cfg["text_config"]
    for key,value in {"hidden_size":2048,"moe_intermediate_size":768,
        "num_experts":128,"num_experts_per_tok":8,"num_hidden_layers":48}.items():
        assert text[key]==value,(key,text[key])
    from safetensors import safe_open
    index=json.loads((Path(MODEL)/"model.safetensors.index.json").read_text())["weight_map"]
    shapes={}
    for key in ["model.language_model.layers.0.mlp.experts.gate_up_proj",
                "model.language_model.layers.0.mlp.experts.down_proj"]:
        with safe_open(str(Path(MODEL)/index[key]),framework="pt") as f:
            shapes[key]=f.get_slice(key).get_shape()
    expected=[[128,2048,1536],[128,768,2048]]
    assert list(shapes.values())==expected,shapes
    result={"model":MODEL,"revision":MODEL_REVISION,"weight_shapes":shapes,
        "dtype":"BF16","upstream_commit":UPSTREAM_COMMIT,
        "torch":torch.__version__,"cuda":torch.version.cuda,
        "vllm":importlib.metadata.version("vllm"),
        "deep_ep":importlib.metadata.version("deep-ep"),
        "gpu":cmd("nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,clocks.sm,temperature.gpu","--format=csv,noheader"),
        "processes":cmd("nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory","--format=csv,noheader"),
        "topology":cmd("nvidia-smi","topo","-m"),
        "git":cmd("git","rev-parse","HEAD"),
        "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path("/home/esjung/MLLM-EP-github/poc_flashvep/reports/native_layered_prefill_ep4_discovery_spec.md"),
            Path("poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json"),
            Path(MODEL)/"config.json"]}}
    atomic_json(Path("poc_native_lp_ep4/artifacts/gpu_audit/environment.json"),result)


if __name__=="__main__":main()
