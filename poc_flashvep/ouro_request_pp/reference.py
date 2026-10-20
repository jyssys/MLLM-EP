"""Actual fixed-R single-GPU forward, not post-hoc selection from R4."""
import torch
from .model_adapter import load_cpu


@torch.inference_mode()
def golden(requests, depth, output_len, stop_eos=False):
    model = load_cpu().to('cuda:0')
    model.model.total_ut_steps = depth
    model.config.total_ut_steps = depth
    counter = [0]
    captured = []
    def hook(module, args, output):
        counter[0] += 1
    handles = [layer.register_forward_hook(hook) for layer in model.model.layers]
    norm_hook = model.model.norm.register_forward_hook(lambda m,a,v: captured.append(v[:, -1:, :].detach().clone()))
    result = {}
    for row in requests:
        input_ids = torch.tensor([row['input_ids']], device='cuda:0')
        past = None
        ids = []
        before = counter[0]
        for _ in range(output_len):
            captured.clear()
            y = model(input_ids, past_key_values=past, use_cache=True, exit_at_step=depth-1, logits_to_keep=1)
            past = y.past_key_values
            token = int(y.logits[0, -1].argmax())
            ids.append(token)
            if stop_eos and token == 0:
                break
            input_ids = torch.tensor([[token]], device='cuda:0')
        assert counter[0]-before == 24*depth*len(ids)
        assert len(past.key_cache) == 24*depth
        result[row['id']] = dict(ids=ids, logits=y.logits.detach().float().cpu(),
            hidden=captured[-1].float().cpu(),
            cache={i:(k.cpu(), past.value_cache[i].cpu()) for i,k in enumerate(past.key_cache)},
            layer_calls=counter[0]-before, finish_reason='eos' if stop_eos and ids[-1]==0 else 'length')
    for h in handles:
        h.remove()
    norm_hook.remove()
    del model, y, past
    torch.cuda.empty_cache()
    return result


def errors(actual, reference):
    delta = (actual.float()-reference.float()).abs()
    return dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
        max_relative=float((delta/reference.float().abs().clamp_min(1e-5)).max()),
        mean_relative=float((delta/reference.float().abs().clamp_min(1e-5)).mean()),
        relative_L2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)),
        bitwise=bool(torch.equal(actual, reference)))
