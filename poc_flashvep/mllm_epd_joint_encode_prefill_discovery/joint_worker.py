"""Instrument owned model calls and time the actual D-local vision-only control."""

import json
import os
from pathlib import Path
import time

from ..mllm_epd_prefill_heavy_modality_discovery.observer import DiscoveryWorker


class JointWorker(DiscoveryWorker):
    def campaign_joint_vision(self, fixture):
        import torch
        from PIL import Image
        from transformers import AutoProcessor

        if os.environ["MLLM_EPD_ROLE"] != "decode":
            raise ValueError("A vision-only D control requires the original D worker")
        contract = Path(os.environ["MLLM_JOINT_CASES"])
        cases = {case["case_id"]: case for case in json.loads(contract.read_bytes())["cases"]}
        selected = cases[fixture]
        paths = [image["path"] for image in selected["request"]["images"][:selected["prefix_images"]]]
        if not hasattr(self, "_joint_cpu_image_processor"):
            from ..mllm_epd_prefill_heavy_token_split.common import ROOT
            model = json.loads((ROOT / "state/model_download_complete.json").read_bytes())["directory"]
            self._joint_cpu_image_processor = AutoProcessor.from_pretrained(model, local_files_only=True).image_processor
        began = time.monotonic_ns()
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        processed = self._joint_cpu_image_processor(images=images, return_tensors="pt")
        pixels, grid = processed["pixel_values"], processed["image_grid_thw"]
        actual_grid = grid.tolist()
        if actual_grid != selected["request"]["grid_thw"][:len(paths)]:
            raise ValueError("Vision-only control differs from frozen image processor grid")
        preprocess_end = time.monotonic_ns()
        visual = self.model_runner.model.visual
        start, finish = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.inference_mode():
            start.record()
            placed = pixels.to(device=visual.device, dtype=visual.dtype)
            result = visual(placed, actual_grid)
            finish.record()
            finish.synchronize()
        return {
            "images": len(paths), "patches": len(pixels), "embedding_rows": len(result),
            "embedding_dtype": str(result.dtype), "embedding_bytes": result.numel() * result.element_size(),
            "cpu_preprocess_ms": (preprocess_end - began) / 1e6,
            "h2d_inclusive_vision_cuda_ms": start.elapsed_time(finish),
            "end_to_end_inside_worker_ms": (time.monotonic_ns() - began) / 1e6,
        }

    def campaign_joint_timing(self, tag):
        """Place native-stream events without resolving them in a measured request."""
        import torch

        runner = self.model_runner
        active = getattr(self, "_joint_timing", None)
        if tag is None:
            if active is None:
                raise ValueError("No active joint timing")
            rows, originals, label = active
            for name, original in originals.items():
                setattr(runner, name, original)
            del self._joint_timing
            torch.cuda.synchronize()
            return {
                "label": label,
                "timeline": [
                    row | {"device_ms": left.elapsed_time(right)}
                    for row, left, right in rows
                ],
            }
        if active is not None:
            raise ValueError("The prior joint timing needs retirement")
        rows = []
        originals = {name: getattr(runner, name) for name in ("_execute_mm_encoder", "_model_forward")}
        for name, original in originals.items():
            def timed(*args, name=name, original=original, **kwargs):
                left, right = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin = time.monotonic_ns()
                left.record()
                result = original(*args, **kwargs)
                right.record()
                rows.append(({"name": name, "host_launch_begin_ns": begin,
                              "host_launch_end_ns": time.monotonic_ns()},
                             left, right))
                return result
            setattr(runner, name, timed)
        self._joint_timing = (rows, originals, tag)
        return {"label": tag, "role": os.environ["MLLM_EPD_ROLE"]}
