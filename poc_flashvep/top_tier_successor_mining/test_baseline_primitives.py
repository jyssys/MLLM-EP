"""Parity/invariant tests against pinned official baseline source.

Small synthetic tensors are implementation tests, never main research evidence.
"""
import argparse
import ast
import json
import os
import types
from pathlib import Path
from gpu_scope import allowed_devices
from typing import Optional

import torch
from torch.utils.cpp_extension import load
from policies import modes_weights, sere_route


def functions_from_source(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError("Official functions not found")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    allowed_devices()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(8422)
    base = args.references / "SERE/vllm/SERE_vllm/rerouting_cuda_ops"
    ext = load(name="sere_successor_reference", sources=[str(base/"rerouting_ops.cpp"),
                str(base/"rerouting_kernel.cu")], extra_cuda_cflags=["-O3"], verbose=True)
    tests = []
    for dtype in [torch.float32, torch.bfloat16]:
        for m in [1, 8, 32, 256]:
            ids = torch.rand(m, 128, device="cuda").topk(8).indices.contiguous()
            weights = torch.rand(m, 8, device="cuda").softmax(-1).contiguous()
            similarity = torch.rand(128,128,device="cuda",dtype=dtype)
            similarity = ((similarity+similarity.T)/2).contiguous()
            similarity.fill_diagonal_(1)
            for retain in [1, 2, 4, 8]:
                for threshold in [0, .5, .9, 1]:
                    ref = ext.reroute(weights, ids.clone(), similarity, retain,
                                     torch.zeros(128,device="cuda",dtype=torch.bool),
                                     torch.zeros(128,device="cuda",dtype=torch.long), threshold)
                    got = sere_route(ids, similarity, retain, threshold)
                    ok = torch.equal(ref, got)
                    tests.append(dict(method="sere", M=m,dtype=str(dtype),retain=retain,
                                      threshold=threshold,exact_ids=ok))
                    assert ok, tests[-1]
    ns = functions_from_source(args.references/"MoDES/models/utils.py",
                                ["apply_scaler_scale"], {"torch": torch})
    ns.update(torch=torch, Optional=Optional, SKIP_EXP_COUNT=0, TOTAL_EXP_COUNT=0)
    ns = functions_from_source(args.references/"MoDES/models/qwen3.py",
                                ["experts_forward", "mlp_forward"], ns)
    # Old official packed matrices have the opposite orientation to HF 5.14.
    E, H, I, M, K = 8, 32, 16, 24, 4
    mlp = types.SimpleNamespace(hidden_size=H, top_k=K,
                               gate=torch.nn.Linear(H,E,bias=False,device="cuda",dtype=torch.bfloat16))
    experts = types.SimpleNamespace(num_experts=E, act_fn=torch.nn.functional.silu,
                    gate_up_proj=torch.randn(E,H,2*I,device="cuda",dtype=torch.bfloat16)/H**.5,
                    down_proj=torch.randn(E,I,H,device="cuda",dtype=torch.bfloat16)/I**.5)
    mlp.experts = lambda h,w,i: ns["experts_forward"](experts,h,w,i)
    mlp.moe_text_mask = (torch.arange(M,device="cuda")%2 == 0)[:,None]
    mlp.moe_media_mask = ~mlp.moe_text_mask
    mlp.gate.moe_text_mask, mlp.gate.moe_media_mask = mlp.moe_text_mask, mlp.moe_media_mask
    mlp.gate.text_layer_importance, mlp.gate.visual_layer_importance = .03, .01
    hidden = torch.randn(1,M,H,device="cuda",dtype=torch.bfloat16)
    for tt,tv in [(0,0),(.002,.002),(.01,.004),(.1,.1)]:
        ref = ns["mlp_forward"](mlp,hidden,enable_tau_skip=True,tau={"text":tt,"visual":tv})
        logits = mlp.gate(hidden.view(-1,H))
        weights, ids = logits.float().softmax(-1).topk(K)
        weights = (weights/weights.sum(-1,keepdim=True)).to(logits.dtype)
        masked, drop = modes_weights(weights, mlp.moe_text_mask[:,0],mlp.moe_media_mask[:,0],
                                     .03,.01,tt,tv)
        dense_weights = torch.zeros_like(logits).scatter_(1,ids,masked)
        got = mlp.experts(hidden.view(-1,H),dense_weights,ids).view_as(hidden)
        tests.append(dict(method="modes",tau_text=tt,tau_vision=tv,
                          max_abs=float((ref-got).abs().max()),exact=torch.equal(ref,got)))
        assert torch.equal(ref,got), tests[-1]
    ablated = ns["mlp_forward"](mlp,hidden,moe_layer_skip_flag=True,skip_modality="text")
    tests.append(dict(method="modes_official_ablation", skipped_output_is_input=
                      torch.equal(ablated.view(-1,H)[mlp.moe_text_mask[:,0]],
                                  hidden.view(-1,H)[mlp.moe_text_mask[:,0]])))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(tests,indent=2))
    print(json.dumps({"tests":len(tests),"status":"PASS","ablation":tests[-1]}),flush=True)


if __name__ == "__main__":
    main()
