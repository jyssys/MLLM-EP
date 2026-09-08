"""CPU-only bridge boundary tests; no claim about full-model GPU parity."""
import json
from types import SimpleNamespace

import torch
from torch import nn

from libra_vl_bridge import install_captured_vl_context


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.rotary_emb = nn.Identity()

    def forward(self, hidden, residual):
        return hidden, residual, "unchanged_planner_metadata"


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(4)])

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return input_embeds


def main():
    torch.manual_seed(7317)
    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        for tokens in (1, 7, 128):
            stack = Stack()
            context = install_captured_vl_context(SimpleNamespace(model=stack))
            dim, hidden_size = 128, 2048
            mask = torch.arange(tokens) % 2 == 0
            context.update(inputs_embeds=torch.randn(tokens, hidden_size).to(dtype),
                           vision_mask=mask,
                           cos=torch.randn(tokens, dim).cos().to(dtype),
                           sin=torch.randn(tokens, dim).sin().to(dtype),
                           deepstack=[torch.randn(int(mask.sum()), hidden_size).to(dtype) for _ in range(3)])
            actual = stack(torch.zeros(tokens, dtype=torch.int64), None, None)
            # Native residual collectives may write into the forward input.
            # Preserve values without aliasing the reusable captured source.
            assert torch.equal(actual, context["inputs_embeds"])
            assert actual.data_ptr() != context["inputs_embeds"].data_ptr()
            q = torch.randn(tokens, 32*dim).to(dtype)
            k = torch.randn(tokens, 4*dim).to(dtype)
            qout, kout = stack.layers[0].self_attn.rotary_emb(None, q, k)
            for x, y, heads in ((q, qout, 32), (k, kout, 4)):
                # Independent reference in HF's [batch, heads, sequence, dim] order.
                v = x.reshape(tokens, heads, dim).permute(1, 0, 2).unsqueeze(0)
                half = torch.cat((-v[..., dim//2:], v[..., :dim//2]), dim=-1)
                expected = v*context["cos"][None, None] + half*context["sin"][None, None]
                expected = expected.squeeze(0).permute(1, 0, 2).reshape_as(x)
                assert torch.equal(expected, y)
            for index, layer in enumerate(stack.layers):
                h = torch.randn(tokens, hidden_size).to(dtype)
                r = torch.randn_like(h)
                output = layer(h, r)
                assert output[2] == "unchanged_planner_metadata"
                if index < 3:
                    expected = h + r
                    expected[mask] += context["deepstack"][index]
                    assert torch.equal(output[0], expected)
                    assert torch.count_nonzero(output[1]) == 0
                else:
                    assert output[0] is h and output[1] is r
            checks.append({"dtype": str(dtype), "tokens": tokens, "status": "PASS"})
    print(json.dumps({"scope": "CPU_BOUNDARY_ONLY_GPU_FULL_MODEL_PARITY_PENDING", "checks": checks}))


if __name__ == "__main__":
    main()
