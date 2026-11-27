"""Frozen Qwen BF16 stage adapters for the EADA research engine."""
import json
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, apply_rotary_pos_emb_vision,
)


def stage_weights(module):
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def packed_vision_forward(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    length = hidden_states.shape[0]
    q,k,v = self.qkv(hidden_states).reshape(length,3,self.num_heads,-1).unbind(1)
    q,k = apply_rotary_pos_emb_vision(q,k,*position_embeddings)
    if max_seqlen is None:
        key = cu_seqlens.data_ptr()
        if key not in self._eada_segments:
            self._eada_segments[key] = int(cu_seqlens.diff().max())
        max_seqlen = self._eada_segments[key]
    attended = flash_attn_varlen_func(q.contiguous(),k.contiguous(),v.contiguous(),
                                       max_seqlen_q=max_seqlen,max_seqlen_k=max_seqlen,
                                       cu_seqlens_q=cu_seqlens,cu_seqlens_k=cu_seqlens,
                                       softmax_scale=self.scaling,causal=False,fa_version=3)
    return self.proj(attended.reshape(length,self.dim))


def install_packed_vision(encoder):
    encoder._eada_segments = {}
    for block in encoder.blocks:
        attn = block.attn
        attn._eada_segments = encoder._eada_segments
        attn._eada_default_forward = attn.forward
        attn.forward = MethodType(packed_vision_forward,attn)


class Checkpoint:
    def __init__(self, path, family):
        self.path, self.family = path, family
        self.vision = family == "qwen-vl"
        self.architecture = Qwen2_5_VLForConditionalGeneration if self.vision else Qwen2ForCausalLM
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True) if self.vision else None
        self.tokenizer = (self.processor.tokenizer if self.vision else
                          AutoTokenizer.from_pretrained(path, local_files_only=True))
        generation = json.loads((path / "generation_config.json").read_text())
        eos = generation.get("eos_token_id", self.tokenizer.eos_token_id)
        self.eos = {eos} if isinstance(eos, int) else set(eos)

    def replica(self, dev, *, vision_device=None, packed_vision=False):
        model = self.architecture.from_pretrained(
            self.path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        ).eval().to(dev)
        if self.vision:
            if vision_device is not None:
                model.model.visual.to(vision_device)
                if packed_vision:
                    install_packed_vision(model.model.visual)
            else:
                del model.model.visual
        return model

    def preprocess(self, row, *, workload):
        if workload == "realistic":
            images = []
            try:
                for image in row["images"]:
                    with Image.open(image["path"]) as original:
                        images.append(original.convert("RGB"))
                batch = self.processor(text=[row["prompt"]], images=images, return_tensors="pt")
            finally:
                for image in images:
                    image.close()
            input_ids = batch["input_ids"]
            with np.load(Path(row["token_metadata_file"]), allow_pickle=False) as metadata:
                expected = metadata["input_ids"].reshape(1, -1)
                if not np.array_equal(input_ids.numpy(), expected):
                    raise ValueError(f"Processor token mismatch: {row['request_id']}")
                positions = torch.from_numpy(metadata["positions"].copy()).reshape(3, 1, -1)
                delta = int(metadata["rope_delta"].reshape(-1)[0])
            if batch["image_grid_thw"].tolist() != row["grid_thw"]:
                raise ValueError(f"Frozen grid mismatch: {row['request_id']}")
            visual = input_ids[0].eq(self.processor.image_token_id).nonzero().flatten()
            if visual.numel() != row["visual_tokens"]:
                raise ValueError(f"Frozen modality mask mismatch: {row['request_id']}")
            return {"ids": input_ids, "positions": positions, "delta": delta, "visual": visual,
                    "pixel": batch["pixel_values"], "grid": batch["image_grid_thw"],
                    "modes": batch["mm_token_type_ids"]}
        if workload == "real-text":
            ids = torch.tensor([self.tokenizer.encode(row["prompt"], add_special_tokens=False)])
            if ids.shape[-1] != row["total_input_tokens"]:
                raise ValueError("Real-text inputs differ from the frozen tokenizer")
            pos = torch.arange(ids.shape[-1]).view(1, 1, -1)
            return {"ids": ids, "positions": pos.expand(3, -1, -1) if self.vision else pos[0],
                    "delta": 0, "visual": torch.empty(0, dtype=torch.long)}
        # Controlled text-only workload at the original multimodal LM token count.
        count = row["total_input_tokens"]
        task = row["question_original"] or row["question"]
        lead = self.tokenizer.encode("The following document is a text-only shape control. "
                                     "Use the information from the context to answer the question.\n")
        passage = self.tokenizer.encode(("The table describes a document collection. "
                                        "Entries indicate the year, the amount, and the document page. "
                                        "Each row contains the source date and one reference number. ") + task)
        end = self.tokenizer.encode("\nQuestion: " + task + "\nAnswer:")
        if count <= len(lead) + len(end) or not passage:
            raise ValueError(f"Insufficient text control context: {row['request_id']}")
        n = count - len(lead) - len(end)
        ids = torch.tensor([lead + (passage * ((n + len(passage)-1)//len(passage)))[:n] + end])
        pos = torch.arange(count).view(1, 1, count)
        return {"ids": ids, "positions": pos.expand(3, -1, -1) if self.vision else pos[0],
                "delta": 0, "visual": torch.empty(0, dtype=torch.long)}

    @staticmethod
    def encode(encoder, item, dev):
        if hasattr(encoder,"_eada_segments"):
            encoder._eada_segments.clear()
        pixels = item.pop("pixel").to(dev, dtype=torch.bfloat16)
        grid = item["grid"].to(dev)
        return encoder(pixels, grid_thw=grid).pooler_output

    def prefill(self, model, item, embeddings, dev, *, verify_rope=False):
        ids = item["ids"].to(dev)
        positions = item["positions"].to(dev)
        text = model.model.language_model if self.vision else model.model
        tokens = text.embed_tokens(ids)
        if embeddings is not None:
            if item["visual"].numel() != embeddings.shape[0]:
                raise ValueError("Vision output and original placeholder count disagree")
            tokens[0].index_copy_(0, item["visual"].to(dev), embeddings.to(dev))
        if verify_rope and self.vision and embeddings is not None:
            native_pos, deltas = model.model.get_rope_index(ids, item["modes"].to(dev), image_grid_thw=item["grid"].to(dev))
            if not torch.equal(native_pos, positions) or int(deltas[0]) != item["delta"]:
                raise ValueError("Native MRoPE differs from the upstream metadata")
        prefix = text(inputs_embeds=tokens, position_ids=positions, use_cache=True)
        logits = model.lm_head(prefix.last_hidden_state[:, -1])
        return prefix.past_key_values, int(logits.argmax(-1)[0])


class TokenDecoder:
    """Original Qwen Full28 layers with the supplied paged FA3 attention."""
    def __init__(self, checkpoint, model, dev):
        self.multimodal = checkpoint.vision
        self.text = model.model.language_model if self.multimodal else model.model
        self.lm_head = model.lm_head
        self.config, self.dev = self.text.config, dev
        self.heads = self.config.num_attention_heads
        self.kv_heads = self.config.num_key_value_heads
        self.dim = self.config.hidden_size // self.heads
        self.n_layers = len(self.text.layers)
        self.rope_section = self.config.rope_parameters.get("mrope_section")
        if self.text.has_sliding_layers:
            raise ValueError("The checkpoint has sliding layers outside this all-causal attention contract")

    def __call__(self, previous, position, state, *, instrument=False):
        batch = len(previous)
        input_ids = torch.tensor(previous, device=self.dev)
        position_ids = torch.tensor(position, device=self.dev).view(batch, 1)
        if self.multimodal:
            position_ids = position_ids[None].expand(3, -1, -1)
        x = self.text.embed_tokens(input_ids[:, None])
        cos, sin = self.text.rotary_emb(x, position_ids)
        measurements = []
        begin = torch.cuda.Event(enable_timing=True) if instrument else None
        if begin:
            begin.record()
        for index, layer in enumerate(self.text.layers):
            if instrument:
                layer_start = torch.cuda.Event(enable_timing=True)
                layer_start.record()
            projected = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(projected).view(batch, 1, self.heads, self.dim).transpose(1, 2)
            k = attn.k_proj(projected).view(batch, 1, self.kv_heads, self.dim).transpose(1, 2)
            v = attn.v_proj(projected).view(batch, self.kv_heads, self.dim)
            if self.multimodal:
                q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_section)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            state.append(index, k[:, :, 0], v)
            if instrument:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
            output = state.attend(index, q[:, :, 0].contiguous())
            if instrument:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            x = x + attn.o_proj(output.reshape_as(x))
            if instrument:
                projected_end = torch.cuda.Event(enable_timing=True)
                projected_end.record()
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            if instrument:
                layer_end = torch.cuda.Event(enable_timing=True)
                layer_end.record()
                measurements.append((layer_start,start,end,projected_end,layer_end))
        logits = self.lm_head(self.text.norm(x)[:, 0])
        sampled = logits.argmax(-1)
        if instrument:
            end_step = torch.cuda.Event(enable_timing=True)
            end_step.record()
        ids = sampled.tolist()
        if instrument:
            labels = ("input_norm_qkv_rope_and_kv_append", "attention", "output_projection_residual", "mlp_norm_residual")
            stages = {name:[row[i].elapsed_time(row[i+1]) for row in measurements]
                      for i,name in enumerate(labels)}
            return ids, logits, {"decode_gpu_enclosure_ms": begin.elapsed_time(end_step),
                                 "attention_gpu_enclosure_ms": sum(stages["attention"]),
                                 "attention_layer_ms": stages["attention"],
                                 "stage_totals_ms":{name:sum(samples) for name,samples in stages.items()},
                                 "last_norm_head_argmax_ms":measurements[-1][-1].elapsed_time(end_step)}
        return ids, logits, None
