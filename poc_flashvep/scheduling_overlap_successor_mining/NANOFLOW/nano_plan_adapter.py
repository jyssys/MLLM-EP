"""Bounded faithful plan-API port for the official Qwen MoE H100 graph.

This is NOT a successor mechanism or the paper's searched optimum. It exposes
the existing splitter/executor with a fixed FFN partition and existing streams.
First probe supports a fixed decode cohort only; shape changes after activation
are rejected rather than silently rebuilding a mutated graph.
"""
import json
import os
from pathlib import Path

TAGS = {
    "LayerNormFFN": "cuda", "Gate": "torch", "FusedMoE": "cutlass",
    "AllReduceFusedMoE": "nccl", "SharedExpertGate": "torch",
    "SharedExpertActivation": "torch", "SharedUG": "torch",
    "SharedActivation": "cuda", "SharedD": "torch", "SharedMul": "torch",
    "AddExperts": "torch", "AddDownBias": "torch",
}

# The native partial-plan loader configures only operators listed in the plan.
# Decode activation inherited these implementations from its initial unsplit
# prefill. A first-use pure-prefill plan must initialize them explicitly.
UNPLANNED_TAGS = {
    "layerNormAttn": "cuda", "kqv": "torch", "kqv_bias": "torch",
    "ropeAppend": "cuda", "decAttn": "batched_cuda", "pfAttn": "batched_cuda",
    "o": "torch",
}


def initialize_unplanned_ops(pipeline):
    for attr, tag in UNPLANNED_TAGS.items():
        op = getattr(pipeline, attr)
        if not hasattr(op, "impl"):
            op.config_tag(tag, {"use_cuda_graph": pipeline.cuda_graph_enabled})


def install():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    from nanoflow.core import CategoryType
    from nanoflow.core.basePipeline import BasePipeline
    from nanoflow.operations import FusedMoE
    from nanoflow.models.qwen2_moe.qwen2_moe_ep import Pipeline
    if getattr(Pipeline, "_successor_plan_adapter", False):
        return
    parts = int(os.environ["SUCCESSOR_NANO_PARTS"])
    assert parts in (1, 2, 4)
    plan_phase = os.environ.get("SUCCESSOR_NANO_PLAN_PHASE", "decode")
    assert plan_phase in {"decode", "prefill"}
    compute_sms = int(os.environ.get("SUCCESSOR_NANO_COMPUTE_SMS", "132"))
    collective_sms = int(os.environ.get("SUCCESSOR_NANO_COLLECTIVE_SMS", "132"))
    supported_sms = set(range(8, 128, 8)) | {132}
    assert compute_sms in supported_sms and collective_sms in supported_sms
    out_dir = Path(os.environ["SUCCESSOR_NANO_PLAN_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Match official LayerNorm/GEMM copy_nano contract; same weight wrappers,
    # same expert partition, same exact routed computation, new IO slices only.
    def copy_moe(self, index):
        copied = FusedMoE(self.name, self.device, nano_idx=index)
        copied.set_category(self.category)
        copied.weights = self.weights
        copied.expand_layer(self.layer_list)
        copied.setShape(self.num_experts, self.moe_intermediate_dim, self.hidden_dim,
                        self.top_k, self.norm_topk_prob, self.ep_size, self.ep_rank)
        self.nano_ops.append(copied)
        return copied

    def categories(self):
        for op in self.original_model_operations:
            op.set_category(CategoryType.MEM if op.name == "AllReduceFusedMoE"
                            else CategoryType.COMP)

    original_update = BasePipeline.update
    original_config_algorithm = Pipeline.config_algorithm

    def config_algorithm(self):
        original_config_algorithm(self)
        if self.auto_search_enabled:
            initialize_unplanned_ops(self)

    def update(self, input_infos, decode_batch_size=0, **kwargs):
        token_count = sum(len(ids) for _, ids in input_infos)
        pure_decode = decode_batch_size == len(input_infos) == token_count
        phase_matches = pure_decode if plan_phase == "decode" else decode_batch_size == 0 and token_count > 0
        active = getattr(self, "_successor_plan_shape", None)
        if active is not None:
            assert phase_matches and token_count == active, "Fixed-plan smoke changed shape or phase"
        elif phase_matches:
            assert token_count >= parts and token_count % parts == 0
            self._successor_plan_shape = token_count
            plan = {}
            for name, tag in TAGS.items():
                plan[name] = {}
                for i in range(parts):
                    name_i = name + str(i) if parts > 1 else name
                    plan[name][name_i] = {
                        "batch_idx": i, "batch_size": token_count // parts,
                        "p_value": collective_sms if name == "AllReduceFusedMoE" else compute_sms,
                        "algo_tag": tag, "extra_dep": [],
                    }
            path = out_dir / f"plan_rank{self.world_rank}.json"
            path.write_text(json.dumps(plan, indent=2))
            self._successor_plan_path = str(path)
        if getattr(self, "_successor_plan_shape", None) is not None:
            kwargs.update(auto_search_enabled=True, nano_split_enabled=parts > 1,
                          profile_result_path=self._successor_plan_path)
        result = original_update(self, input_infos, decode_batch_size, **kwargs)
        if getattr(self, "_successor_plan_shape", None) is not None and not getattr(self, "_plan_saved", False):
            records = [{"name": op.name, "M": op.batch_size,
                        "sm_count": op.sm_count,
                        "stream": int(op.stream.cuda_stream),
                        "category": str(op.category), "kernel_tag": op.tag}
                       for op in self.model_operations if op.original_name in TAGS]
            (out_dir / f"active_rank{self.world_rank}.json").write_text(json.dumps({
                "parts": parts, "M": token_count, "operations": records,
                "fixed_phase": plan_phase,
                "compute_sms": compute_sms, "collective_sms": collective_sms,
                "cross_category_disjoint_sms_verified": False,
                "ordered_operations": self.executor.ordered_operations,
                "scope": "manual fixed-plan official splitter/executor; not searched optimum",
            }, indent=2))
            self._plan_saved = True
        return result

    FusedMoE.copy_nano = copy_moe
    # Existing splitter still uses this pre-refactor spelling.
    BasePipeline.is_auto_search_enabled = property(lambda self: self.auto_search_enabled)
    Pipeline.init_category = categories
    Pipeline.config_algorithm = config_algorithm
    Pipeline.update = update
    Pipeline._successor_plan_adapter = True


def worker_with_plan(T0, rank, affinity_module_path, *rest):
    install()
    from nanoflow.entry.worker_entry import worker_entry
    return worker_entry(T0, rank, affinity_module_path, *rest)
