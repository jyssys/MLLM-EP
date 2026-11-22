"""Run an explicitly diagnostic FP32 original-model attention reference."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import torch
from transformers import AutoModelForCausalLM
from .common import ROOT, MODEL, MODEL_REVISION, write, digest


def main():
    folder = Path(sys.argv[1])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    row = json.loads((ROOT / 'data/qualification8.json').read_text())[0]
    model, loading = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys'), loading
    loading = {k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()}
    assert model.config.num_hidden_layers == 36
    model.eval().to('cuda')
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    ids = torch.tensor([row['input_ids']['qwen8']], device='cuda', dtype=torch.long)
    assert ids.shape[1] == row['input_tokens']
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        result = model.model(input_ids=ids, use_cache=True, return_dict=True)
        hidden = result.last_hidden_state[0, -1].detach()
        logits = model.lm_head(hidden[None])[0].float()
    cache = result.past_key_values
    positions = torch.linspace(0, ids.shape[1] - 1, 32, device='cuda').round().long().unique()
    planes = []
    for i in range(36):
        if hasattr(cache, 'layers'):
            key, value = cache.layers[i].keys, cache.layers[i].values
        else:
            key, value = cache.key_cache[i], cache.value_cache[i]
        key = key[0, :, positions, :].permute(1, 0, 2)
        value = value[0, :, positions, :].permute(1, 0, 2)
        planes.append(torch.cat((key, value), dim=-1).cpu())
    torch.save(dict(hidden_last=hidden.cpu(), logits=logits.cpu(), KV_samples=torch.stack(planes),
                    positions=positions.cpu()), folder / 'fp32_reference.pt')
    torch.save(dict(full_prompt_hidden=result.last_hidden_state[0].detach().cpu()),
               folder / 'fp32_full_prompt_hidden.pt')
    top = torch.topk(logits, 8)
    comparisons = []
    # Both are observed BF16 D last-query hidden tensors. Their natural model math is not changed here.
    for name in ('Q22_NativeValidPrefix_N8_FullState_v2', 'Q30_CompiledNative_Original0_N1_KernelPath_v1'):
        path = ROOT / 'raw' / name / 'prompt_state_decode.json'
        if not path.exists():
            continue
        state = json.loads(path.read_text())['requests']['q0']
        measured = torch.load(state['hidden_tensor_file'], map_location='cpu', weights_only=True)['hidden_last'].flatten().float().cuda()
        delta = measured - hidden
        with torch.inference_mode():
            common_head = model.lm_head(measured[None])[0].float()
        comparisons.append(dict(name=name, hidden_max_absolute_error=float(delta.abs().max()),
            hidden_RMS_error=float(delta.square().mean().sqrt()),
            hidden_relative_RMS_error=float(delta.square().mean().sqrt() / hidden.square().mean().sqrt()),
            hidden_cosine=float(torch.nn.functional.cosine_similarity(measured[None], hidden[None])),
            common_FP32_head_argmax=int(common_head.argmax()),
            scope='Common FP32 LM head applied to measured BF16 hidden. This is not the actual runtime BF16 logit trace.'))
    torch.cuda.synchronize()
    write(folder / 'result.json', dict(time=time.time(), passed=True, model_revision=MODEL_REVISION,
        prompt_id=row['prompt_id'], input_tokens=row['input_tokens'],
        dataset_SHA256=digest(ROOT / 'data/qualification8.json'),
        parameter_dtype='FP32 diagnostic conversion of the original checkpoint',
        loading=loading, attention='PyTorch SDPA MATH,exact causal mask,TF32 disabled',
        model_forward_source_SHA256=hashlib.sha256(inspect.getsource(type(model.model).forward).encode()).hexdigest(),
        GPU_UUID=str(torch.cuda.get_device_properties(0).uuid),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        FP32_top8_ids=top.indices.cpu().tolist(), FP32_top8_logits=top.values.cpu().tolist(),
        FP32_top1_top2_margin=float(top.values[0] - top.values[1]), comparisons=comparisons,
        tensor_SHA256=digest(folder / 'fp32_reference.pt'), evidence='DIAGNOSTIC',
        scope='Original Transformers model and checkpoint. Diagnostic high-precision reference only. '
            'No greedy outputs are overwritten. No FP32 time is a BF16 baseline or serving speedup. '
            'Last-prompt query geometry differs from the Native D recomputation. '
            'Small errors do not by themselves identify a specific compiled kernel or prove all requests correct.'))
    print('FP32_ORIGINAL_REFERENCE_COMPLETE', comparisons, flush=True)


if __name__ == '__main__':
    main()
