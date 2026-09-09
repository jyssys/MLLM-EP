"""Same-prefix numerical diagnostic; never timing evidence.

Capture rank-zero vocabulary logits after the native executor has completed.
The original worker, sampler, numerical operations and route selection remain.
"""
import os
from pathlib import Path


def install():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    import torch
    from nanoflow.models.qwen2_moe.qwen2_moe_ep import Pipeline
    original = Pipeline.run
    count = int(os.environ["SUCCESSOR_NANO_COHORT_COUNT"])
    output = Path(os.environ["SUCCESSOR_NANO_LOGITS_OUT"])
    output.mkdir(parents=True, exist_ok=True)

    def run(self):
        result = original(self)
        if self.world_rank == 0 and self.input_req_idx and min(self.input_req_idx) >= 2 * count:
            index = getattr(self, "_successor_logit_step", 0)
            logits = self.sample.children[-1].inputs["logits"].tensor
            positions = [int(i) - 1 for i in self.cumsum_input[1:]]
            # Original run already waits for the executor and copies token IDs.
            # Additional D2H is diagnostic overhead, explicitly excluded.
            captured = logits[positions].detach().cpu().clone()
            torch.save({"step": index, "request_indices": list(self.input_req_idx),
                        "logits": captured, "generated": result,
                        "native_logits_shape": list(logits.shape),
                        "total_input_tokens": int(self.cumsum_input[-1]),
                        "selected_last_token_positions": positions,
                        "timing_excluded": True}, output / f"step{index:03d}.pt")
            self._successor_logit_step = index + 1
        return result

    Pipeline.run = run


def worker_with_checks(T0, rank, affinity_module_path, *rest):
    if "SUCCESSOR_NANO_PARTS" in os.environ:
        from nano_plan_adapter import install as plan_install
        plan_install()
    install()
    from nanoflow.entry.worker_entry import worker_entry
    return worker_entry(T0, rank, affinity_module_path, *rest)
