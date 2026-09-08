"""CPU regression at the actual packed-checkpoint -> native Linear seam."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from libra_vl_bridge import translated_vl_weights


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((args.model/"config.json").read_text())["text_config"]
    index=json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    base="model.language_model.layers.0.mlp.experts."
    raw={}
    for suffix in ("down_proj","gate_up_proj"):
        key=base+suffix
        with safe_open(args.model/index[key],framework="pt",device="cpu") as f:
            raw[key]=f.get_slice(key)[0:1].float()
    translated=dict(translated_vl_weights(raw.items()))
    h=config["hidden_size"];m=config["moe_intermediate_size"]
    prefix="model.layers.0.mlp.experts.0."
    # Exercise the consumer's copy shape, not a permissive reshape.
    down=torch.empty(h,m);down.copy_(translated[prefix+"down_proj.weight"])
    gate=torch.empty(m,h);gate.copy_(translated[prefix+"gate_proj.weight"])
    up=torch.empty(m,h);up.copy_(translated[prefix+"up_proj.weight"])
    torch.manual_seed(8173)
    x=torch.randn(3,h)
    packed_gate,packed_up=(x @ raw[base+"gate_up_proj"][0]).chunk(2,dim=-1)
    expected=(torch.nn.functional.silu(packed_gate)*packed_up) @ raw[base+"down_proj"][0]
    actual=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,gate))*
                                      torch.nn.functional.linear(x,up),down)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    print(json.dumps({"status":"PASS","scope":"REAL_CHECKPOINT_CPU_WEIGHT_LAYOUT_AND_EXPERT_MATH",
                      "max_abs":float((actual-expected).abs().max()),"H":h,"I":m}))


if __name__=="__main__":main()
