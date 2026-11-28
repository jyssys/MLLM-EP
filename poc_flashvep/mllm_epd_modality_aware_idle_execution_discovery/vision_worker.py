"""Physical execution of stock Vision modules already resident on owned E/D workers."""

import itertools
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from ..mllm_epd_prefill_heavy_modality_discovery.observer import DiscoveryWorker


class VisionSession:
    def __init__(self, runner):
        self.vision = runner.model.visual
        self.cut_layer = min(self.vision.fullatt_block_indexes)
        if self.cut_layer != 7 or len(self.vision.blocks) != 32 or self.vision.hidden_size != 1280:
            raise ValueError("Expected the pinned 32-block Qwen2.5-VL vision graph")
        self.root = Path(os.environ["MLLM_IDLE_FIXTURES"]).resolve(strict=True)
        self.loaded = {}
        self.groups = {}
        self.references = {}
        self.held = {}
        self.join = None

    def load(self, name):
        import torch
        from safetensors.torch import load_file

        if not re.fullmatch(r"(?:image_[0-5]|group_(?:left|right|all))", name):
            raise ValueError("Unknown frozen image fixture")
        if name not in self.loaded:
            value = load_file(str(self.root/f"{name}.safetensors"))
            grid = value["grid"].tolist()
            pixels = value["pixel_values"].pin_memory()
            meta = self.vision.prepare_encoder_metadata(grid)
            lengths = []
            for t,h,w in grid:
                _, ends = self.vision.get_window_index_thw(t,h,w)
                offsets = sorted(set([0, *ends.tolist()]))
                lengths.extend(b-a for a,b in itertools.pairwise(offsets))
            if sum(lengths) != len(pixels):
                raise ValueError("Window lengths do not cover original patches")
            self.loaded[name] = (pixels, grid, meta, lengths,
                                 pixels.to(device=self.vision.device, dtype=self.vision.dtype))
        pixels, grid, _, lengths, _ = self.loaded[name]
        return {"role": os.environ["MLLM_EPD_ROLE"], "pid": os.getpid(), "grid": grid,
                "backend": str(self.vision.attn_backend), "dtype": str(self.vision.dtype),
                "weights_bytes": sum(p.numel()*p.element_size() for p in self.vision.parameters()),
                "cut_layer": self.cut_layer, "window_rows": lengths,
                "patch_rows": int(pixels.shape[0]), "processor_bytes": pixels.numel()*pixels.element_size()}

    def timed(self, operation):
        import torch

        left = torch.cuda.Event(enable_timing=True)
        right = torch.cuda.Event(enable_timing=True)
        host = time.monotonic()
        left.record()
        result = operation()
        right.record()
        right.synchronize()
        return result, {"gpu_ms": left.elapsed_time(right), "host_ms": 1000*(time.monotonic()-host)}

    def send(self, tensor):
        from torch.multiprocessing.reductions import reduce_tensor
        from .vision_ipc import encode

        if len(self.held) >= 3:
            raise RuntimeError("Previous IPC owner not yet retired")
        ticket = uuid4().hex
        wire = tensor.reshape(-1,tensor.shape[-1]).contiguous()
        fn, args = reduce_tensor(wire)
        if fn.__name__ != "rebuild_cuda_tensor":
            raise RuntimeError("The export is not physical CUDA IPC")
        self.held[ticket] = wire
        return {"ticket": ticket, "layout": list(tensor.shape), "bytes": wire.numel()*wire.element_size(),
                "args": encode(args)}

    def receive(self, packet):
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        from .vision_ipc import decode

        view = rebuild_cuda_tensor(*decode(packet["args"]))
        shape = packet["layout"]
        if shape not in ([view.shape[0],view.shape[1]],[view.shape[0],1,view.shape[1]]):
            raise RuntimeError("CUDA IPC view layout drift")
        return view.clone().view(shape)

    def prefix(self, value, meta, lengths, first, last):
        import numpy as np
        import torch
        from vllm.model_executor.models.qwen2_5_vl import MMEncoderAttention

        vision = self.vision
        first_row, end_row = sum(lengths[:first]), sum(lengths[:last])
        if len(value) != end_row-first_row:
            raise ValueError("Invalid window row slice")
        subset = lengths[first:last]
        host_cu = np.array([0, *itertools.accumulate(subset)], dtype=np.int32)
        cu = MMEncoderAttention.maybe_recompute_cu_seqlens(
            vision.attn_backend, host_cu, vision.hidden_size, vision.tp_size, vision.device,
            fp8_padded_hidden_size=vision.fp8_padded_hidden_size)
        seq = MMEncoderAttention.maybe_compute_seq_lens(vision.attn_backend, host_cu, vision.device)
        maxseq = torch.tensor(MMEncoderAttention.compute_max_seqlen(vision.attn_backend, host_cu),
                              dtype=torch.int32)
        for block in vision.blocks[:self.cut_layer]:
            value = block(value, cu_seqlens=cu, max_seqlen=maxseq, sequence_lengths=seq,
                          rotary_pos_emb_cos=meta["rotary_pos_emb_cos"][first_row:end_row],
                          rotary_pos_emb_sin=meta["rotary_pos_emb_sin"][first_row:end_row])
        return value

    def patch(self, device_pixels, meta):
        vision = self.vision
        value = vision.patch_embed(device_pixels)
        unit = vision.spatial_merge_unit
        return value.view(-1,unit,vision.hidden_size)[meta["window_index"]].reshape(
            -1,1,vision.hidden_size).contiguous()

    def finish(self, value, meta):
        vision = self.vision
        for i in range(self.cut_layer,len(vision.blocks)):
            shape = "full" if i in vision.fullatt_block_indexes else "window"
            offsets = meta["cu_seqlens" if shape == "full" else "cu_window_seqlens"]
            value = vision.blocks[i](
                value, cu_seqlens=offsets, max_seqlen=meta[f"max_seqlen_{shape}"],
                sequence_lengths=meta[f"sequence_lengths_{shape}"],
                rotary_pos_emb_cos=meta["rotary_pos_emb_cos"], rotary_pos_emb_sin=meta["rotary_pos_emb_sin"])
        return vision.merger(value)[meta["reverse_indices"], :].contiguous()

    def compare(self, name, result):
        import torch

        expected = self.references[name]
        if result.shape != expected.shape or result.dtype != expected.dtype:
            raise RuntimeError("Original encoder output layout changed")
        a,b = expected.float(), result.float()
        error = (a-b).abs()
        cosine = torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)
        return {"output_rows": int(result.shape[0]), "output_width": int(result.shape[1]),
                "bitwise": bool(torch.equal(expected,result)), "max_abs": float(error.max()),
                "different_values": int((error!=0).sum()), "relative_l2": float(
                    error.norm()/a.norm().clamp_min(1e-20)), "cosine": float(cosine)}

    def call(self, form):
        import torch

        op = form["op"]
        if op == "release":
            del self.held[form["ticket"]]
            return {"ack_and_retired": True}
        name = form["fixture"]
        if op == "load":
            return self.load(name)
        pixels, grid, meta, lengths, device = self.loaded[name]
        if op == "balanced_group":
            names = tuple(form["members"])
            if not names or len(names) != len(set(names)) or any(item not in self.loaded for item in names):
                raise ValueError("A group must use distinct loaded original images")
            if names not in self.groups:
                grouped_pixels = torch.cat([self.loaded[item][0] for item in names]).pin_memory()
                grids = [self.loaded[item][1][0] for item in names]
                self.groups[names] = grouped_pixels, grids, self.vision.prepare_encoder_metadata(grids)
            grouped_pixels, grids, metadata = self.groups[names]
            data, upload = self.timed(lambda: grouped_pixels.to(device=self.vision.device,
                                        dtype=self.vision.dtype, non_blocking=True))
            value, encode = self.timed(lambda: self.vision(data, grids, encoder_metadata=metadata))
            if form.get("wire"):
                return {"upload": upload, "vision": encode, "packet": self.send(value)}
            self.kept_images = (names, value)
            return {"upload": upload, "vision": encode}
        if op == "join_images":
            left_names, left = self.kept_images
            right_names = tuple(form["remote_members"])
            ordered = tuple(form["original_order"])
            if sorted(left_names+right_names) != sorted(ordered) or len(set(ordered)) != len(ordered):
                raise ValueError("The assignments do not partition original images")
            right, copy = self.timed(lambda: self.receive(form["packet"]))
            def sizes(members):
                return [len(self.loaded[item][0])//self.vision.spatial_merge_unit for item in members]
            left_rows = dict(zip(left_names,left.split(sizes(left_names)),strict=True))
            right_rows = dict(zip(right_names,right.split(sizes(right_names)),strict=True))
            packed, gather = self.timed(lambda: torch.cat([(left_rows|right_rows)[item] for item in ordered]))
            self.kept_images = None
            return {"return_ipc":copy,"gather":gather,"check":self.compare(name,packed)}
        if op in ("native","full_send"):
            if op == "native":
                value, times = self.timed(lambda: self.vision(device,grid,encoder_metadata=meta))
                if form.get("reference"):
                    self.references[name] = value.clone()
                result = self.compare(name,value) if name in self.references else {}
                return {"vision": times, "check": result}
            placed, copy = self.timed(lambda: pixels.to(device=self.vision.device,
                                                          dtype=self.vision.dtype, non_blocking=True))
            value, times = self.timed(lambda: self.vision(placed,grid,encoder_metadata=meta))
            return {"input_h2d": copy, "vision": times, "packet": self.send(value)}
        if op == "image_output":
            value, copy = self.timed(lambda: self.receive(form["packet"]))
            return {"return_ipc": copy, "check": self.compare(name,value)}
        cut = form.get("cut")
        if not isinstance(cut,int) or not 0<cut<len(lengths):
            raise ValueError("Split requires two nonempty sets of complete windows")
        boundary = sum(lengths[:cut])
        if op == "prepare":
            if self.join is not None:
                raise RuntimeError("A live vision trial has not passed the join")
            states, patch = self.timed(lambda: self.patch(device,meta))
            self.join = (name,cut,states[:boundary].contiguous(),None)
            return {"patch": patch, "packet": self.send(states[boundary:].contiguous()),
                    "rows_kept": boundary, "rows_sent": len(states)-boundary}
        if op == "kept":
            saved, split, initial, final = self.join
            if (saved,split)!=(name,cut) or final is not None:
                raise RuntimeError("Kept packet identity drift")
            value, compute = self.timed(lambda: self.prefix(initial,meta,lengths,0,cut))
            self.join = (name,cut,initial,value)
            return {"prefix": compute}
        if op == "remote":
            initial, copy = self.timed(lambda: self.receive(form["packet"]))
            value, compute = self.timed(lambda: self.prefix(initial,meta,lengths,cut,len(lengths)))
            return {"input_ipc": copy, "prefix": compute, "packet": self.send(value)}
        if op == "join":
            saved, split, _, kept = self.join
            if (saved,split)!=(name,cut) or kept is None:
                raise RuntimeError("Cannot join a missing or unrelated window packet")
            arrived, copy = self.timed(lambda: self.receive(form["packet"]))
            value, tail = self.timed(lambda: self.finish(torch.cat((kept,arrived)),meta))
            self.join = None
            return {"return_ipc": copy, "tail": tail, "check": self.compare(name,value)}
        if op in ("local_same_cut","local_frontier"):
            def execute():
                value = self.patch(device,meta)
                if op=="local_frontier":
                    value = self.prefix(value,meta,lengths,0,len(lengths))
                else:
                    a = self.prefix(value[:boundary],meta,lengths,0,cut)
                    b = self.prefix(value[boundary:],meta,lengths,cut,len(lengths))
                    value = torch.cat((a,b))
                return self.finish(value,meta)
            value, compute = self.timed(execute)
            return {"vision":compute,"check":self.compare(name,value)}
        raise ValueError(f"Unrecognized physical operation {op}")


class IdleWorker(DiscoveryWorker):
    def campaign_idle_vision(self, body):
        import torch

        if os.environ["MLLM_EPD_ROLE"] not in ("encoder","decode"):
            raise ValueError("Only the owned E and D workers run these probes")
        if not hasattr(self, "_idle_vision"):
            self._idle_vision = VisionSession(self.model_runner)
        with torch.inference_mode():
            return self._idle_vision.call(body)

    def campaign_profile_begin(self, name, include_copies=True, state_interval_ms=0, buffer_gib=1):
        if os.environ["MLLM_EPD_ROLE"]=="decode" and os.environ.get("MLLM_IDLE_GRAPH")=="1":
            from ..mllm_epd_prefill_heavy_token_split.common import ROOT
            from .graph_observer import GraphRecorder

            self._campaign_collector = GraphRecorder(name, run_dir=ROOT)
            return self._campaign_collector.begin()
        return super().campaign_profile_begin(name, include_copies, state_interval_ms, buffer_gib)
