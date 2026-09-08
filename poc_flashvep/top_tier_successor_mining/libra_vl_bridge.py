"""Bounded Qwen-VL text-stack bridge to the supplied Libra implementation.

Original Libra planning, expert kernels, streams and weight-copy algorithm stay
unchanged. This bridge only translates packed checkpoint names and supplies
captured exact VL embeddings/MRoPE/DeepStack to the native decoder. It is a
prefill diagnostic, NOT a production multimodal serving integration. Numerical
parity to the full HF model is mandatory before interpreting timings.
"""
import json
from pathlib import Path


def checkpoint_descriptor(source, destination):
    """No weight copy/rewrite: symlinks plus an explicit text-config descriptor."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root_config = json.loads((source / "config.json").read_text())
    config = root_config["text_config"]
    config.update(model_type="qwen3_moe", architectures=["Qwen3MoeForCausalLM"],
                  torch_dtype="bfloat16",tie_word_embeddings=root_config["tie_word_embeddings"])
    # Captured exact MRoPE cos/sin replaces the native RoPE module before use.
    # An ordinary constructor is needed solely to allocate the native decoder.
    config["rope_scaling"] = None
    config["rope_theta"] = config.get("rope_theta",config.get("rope_parameters", {}).get("rope_theta",1000000))
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    for path in source.iterdir():
        if path.name == "config.json" or path.is_dir():
            continue
        target = destination / path.name
        if not target.exists():
            target.symlink_to(path)
    (destination / "BRIDGE_PROVENANCE.json").write_text(json.dumps({
        "source_checkpoint":str(source),"kind":"TEXT_CONFIG_DESCRIPTOR_ONLY",
        "weights":"Unmodified source safetensors via symlinks",
        "required_runtime_bridge":__file__},indent=2))
    return destination


def translated_vl_weights(weights):
    for name, tensor in weights:
        if name.startswith("model.visual."):
            continue
        name = name.replace("model.language_model.", "model.")
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for expert, value in enumerate(tensor):
                # Qwen-VL checkpoint stores [H,2I] for x @ W. Native SG
                # Linear loaders expect [I,H] separately, not packed [2I,H].
                gate, up = value.chunk(2, dim=-1)
                yield f"{prefix}.{expert}.gate_proj.weight", gate.T.contiguous()
                yield f"{prefix}.{expert}.up_proj.weight", up.T.contiguous()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for expert, value in enumerate(tensor):
                yield f"{prefix}.{expert}.down_proj.weight", value.T.contiguous()
        else:
            yield name, tensor


def install_weight_name_bridge():
    from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
    original = Qwen3MoeForCausalLM.load_weights
    def load(self, weights):
        return original(self, translated_vl_weights(weights))
    Qwen3MoeForCausalLM.load_weights = load


def install_captured_vl_context(model):
    import torch
    from torch import nn
    context = {}
    class ExactMRoPE(nn.Module):
        def forward(self, positions, query, key):
            cos, sin = context["cos"], context["sin"]
            dim = cos.shape[-1]
            def rotate(x):
                original_shape = x.shape
                x = x.view(x.shape[0], -1, dim)
                first, second = x.chunk(2, dim=-1)
                half = torch.cat((-second, first), dim=-1)
                return (x * cos[:,None,:] + half * sin[:,None,:]).reshape(original_shape)
            return rotate(query), rotate(key)
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = ExactMRoPE()
    original = model.model.forward
    def forward(input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        assert input_ids.numel() == context["inputs_embeds"].shape[0]
        return original(input_ids, positions, forward_batch,
                        # Native layer0 residual aliases this storage and the
                        # DP scatter overwrites it. Stock embedding lookup is
                        # fresh each forward; preserve that lifetime contract.
                        input_embeds=context["inputs_embeds"].clone(), **kwargs)
    model.model.forward = forward
    for index, layer in enumerate(model.model.layers):
        if index >= 3:
            break
        def add_deepstack(module, args, output, index=index):
            hidden, residual, *rest = output
            # HF adds DeepStack to the already residual-combined block output.
            # Make that arithmetic order explicit; do not add it to the MoE
            # branch alone. Carry a zero residual to the next native block.
            hidden = hidden + residual
            hidden = hidden.clone()
            mask = context["vision_mask"]
            hidden[mask] += context["deepstack"][index]
            return (hidden, torch.zeros_like(residual), *rest)
        layer.register_forward_hook(add_deepstack)
    return context


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--out",required=True)
    args = parser.parse_args()
    print(checkpoint_descriptor(args.source,args.out))
