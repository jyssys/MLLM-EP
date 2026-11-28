"""Check group-wise cross-KV projection with the stock Transformers Mllama model on CPU."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
from transformers import MllamaTextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.mllama.modeling_mllama import (
    MllamaForCausalLM,
    _prepare_cross_attention_mask,
)
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)


def project_cross_kv(attention, group):
    def to_heads(projected):
        return projected.reshape(
            group.shape[0], group.shape[1], attention.num_key_value_heads, attention.head_dim
        ).transpose(1, 2)

    keys = attention.k_norm(to_heads(attention.k_proj(group)))
    return keys, to_heads(attention.v_proj(group))


def compare(expected, actual):
    return {
        "bitwise": torch.equal(expected, actual),
        "max_abs": (expected.float() - actual.float()).abs().max().item(),
    }


def processor_visibility(ids, image_token):
    sparse = [get_cross_attention_token_mask(ids[0].tolist(), image_token)]
    return torch.from_numpy(
        convert_sparse_cross_attention_mask_to_dense(sparse, [[1, 1, 1]], max_num_tiles=1, length=ids.shape[1])
    )


@torch.inference_mode()
def probe(seed, dtype):
    torch.manual_seed(seed)
    config = MllamaTextConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, cross_attention_layers=[1, 3],
        vocab_size=97, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = MllamaForCausalLM(config).eval().to(device="cpu", dtype=dtype)
    for index in config.cross_attention_layers:
        layer = model.model.layers[index]
        layer.cross_attn_attn_gate.fill_(0.7)
        layer.cross_attn_mlp_gate.fill_(0.4)

    visual_groups = [torch.randn(1, 5, config.hidden_size, dtype=dtype) for _ in range(3)]
    image_token = config.vocab_size - 1
    text = torch.randint(3, image_token, (1, 9))
    ids = text.clone()
    ids[:, ::3] = image_token
    visibility = processor_visibility(ids, image_token)

    def forward(tokens, mask, *, features=None, cache=None):
        attention_mask, row_mask = _prepare_cross_attention_mask(mask, num_vision_tokens=5, dtype=dtype)
        return model(
            input_ids=tokens, cross_attention_states=features, cross_attention_mask=attention_mask,
            full_text_row_masked_out_mask=row_mask, past_key_values=cache, use_cache=True,
        )

    complete_features = torch.cat(visual_groups, dim=1)
    native = forward(ids, visibility, features=complete_features)
    adjacent = text.clone()
    adjacent[:, :3] = image_token
    adjacent_visibility = processor_visibility(adjacent, image_token)
    if not adjacent_visibility[:, :3, 1:].any():
        raise RuntimeError("The negative control must omit a visible image group.")
    adjacent_native = forward(adjacent, adjacent_visibility, features=complete_features)
    premature = forward(adjacent[:, :3], adjacent_visibility[:, :3, :1], features=visual_groups[0])

    phased_logits = []
    cache = DynamicCache(config=config)
    for group_index, group in enumerate(visual_groups):
        for layer_index in config.cross_attention_layers:
            attention = model.model.layers[layer_index].cross_attn
            keys, values = project_cross_kv(attention, group)
            cache.update(keys, values, layer_index)
        start = 3 * group_index
        if visibility[:, start:start + 3, group_index + 1:].any():
            raise RuntimeError("A query in this text group can see a visual state that is still missing.")
        result = forward(ids[:, start:start + 3], visibility[:, start:start + 3, :group_index + 1], cache=cache)
        phased_logits.append(result.logits)

    before_decode_cache = [
        {
            "layer": index, "kind": "cross" if index in config.cross_attention_layers else "self",
            "keys": compare(reference.keys, actual.keys), "values": compare(reference.values, actual.values),
        }
        for index, (reference, actual) in enumerate(zip(native.past_key_values.layers, cache.layers, strict=True))
    ]
    merged_logits = torch.cat(phased_logits, dim=1)
    teacher_ids = torch.randint(3, image_token, (1, 4))
    reference_decode, candidate_decode = [], []
    for step in range(teacher_ids.shape[1]):
        tokens = teacher_ids[:, step:step + 1]
        reference_decode.append(forward(tokens, visibility[:, -1:], cache=native.past_key_values).logits)
        candidate_decode.append(forward(tokens, visibility[:, -1:], cache=cache).logits)
    expected_post = torch.cat(reference_decode, dim=1)
    actual_post = torch.cat(candidate_decode, dim=1)

    return {
        "seed": seed, "dtype": str(dtype),
        "prefill": compare(native.logits, merged_logits),
        "prefill_greedy_matches": int((native.logits.argmax(-1) == merged_logits.argmax(-1)).sum()),
        "teacher_forced_decode": compare(expected_post, actual_post),
        "teacher_forced_greedy_matches": int((expected_post.argmax(-1) == actual_post.argmax(-1)).sum()),
        "denominators": {"prefill_rows": ids.shape[1], "teacher_decode_rows": teacher_ids.shape[1]},
        "before_decode_cache": before_decode_cache,
        "negative_control_adjacent_image_markers": compare(adjacent_native.logits[:, :3], premature.logits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = Path(inspect.getsourcefile(MllamaForCausalLM))
    processor = Path(inspect.getsourcefile(get_cross_attention_token_mask))
    record = {
        "evidence": "CORRECTNESS_ONLY",
        "workload": "CPU with synthetic three-image hidden states and initialized tiny Mllama text weights",
        "architecture": {"layers": 4, "cross_layers": [1, 3], "model_width": 64, "visual_rows_per_group": 5},
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "mllama_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mllama_processor_source_sha256": hashlib.sha256(processor.read_bytes()).hexdigest(),
        "results": [probe(seed, dtype) for dtype in (torch.float32, torch.bfloat16) for seed in (1, 2, 3, 4)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
