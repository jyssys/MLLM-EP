"""Explicit evidence-only worker extension. The default worker is unchanged."""

from poc_ouro_parallelism_deep_discovery.operand_binding import (
    LiftedOperandBindingCapture, ROLES, expected_weight_bindings, parameter_snapshot,
)
from poc_ouro_parallelism_deep_discovery.worker_probe import CampaignProbe


class OperandBindingProbe(CampaignProbe):
    def start_operand_binding_capture(self):
        """Arm metadata capture after compilation and loading finish."""
        import torch
        from torch._subclasses.fake_tensor import FakeTensor
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if hasattr(self, "_operand_binding_capture"):
            raise RuntimeError("This worker already has an operand capture.")
        model = self.get_model()
        core = model.model
        aot = getattr(core, "aot_compiled_fn", None)
        if aot is None:
            raise RuntimeError("The AOT callable does not exist. Complete warmup first.")
        owner = aot._artifacts.compiled_fn
        if getattr(owner, "_fake_mode", None) is not None:
            raise RuntimeError("The AOT backend did not finish loading.")
        graph = getattr(owner.graph_module, "graph", None)
        if graph is None:
            raise RuntimeError("The loaded callable has no inspectable FX graph.")
        placeholders = [str(node.target) for node in graph.nodes if node.op == "placeholder"]
        weights = [name for name in placeholders if "parameters_weight" in name.lower()]
        mode = core._campaign_storage_evidence["mode"]
        layers = len(core.layers)
        expected = expected_weight_bindings(placeholders, mode, layers)
        parameters = parameter_snapshot(model.named_parameters(remove_duplicate=False))
        storage_groups = []
        for layer in range(layers):
            for role, path in ROLES.items():
                names = [f"model._campaign_execution.{depth}.{layer}.{path}.weight"
                         for depth in range(4)]
                pointers = [parameters[name]["storage_data_ptr"] for name in names]
                expected_unique = 1 if mode == "alias" else 4
                if len(set(pointers)) != expected_unique:
                    raise RuntimeError("Execution-module storage ownership does not match the requested arm.")
                storage_groups.append({"layer": layer, "role": role,
                                       "parameter_names": names,
                                       "storage_data_ptr_by_depth": pointers})

        def context():
            if not is_forward_context_available():
                return {"forward_context_available": False}
            forward = get_forward_context()
            return {"forward_context_available": True,
                    "cudagraph_runtime_mode": forward.cudagraph_runtime_mode.name,
                    "batch_descriptor": repr(forward.batch_descriptor)}

        capture = LiftedOperandBindingCapture(
            owner, placeholders, parameters, expected,
            is_real_tensor=lambda value: isinstance(value, torch.Tensor)
            and not isinstance(value, FakeTensor) and not value.is_meta,
            context_provider=context,
        )
        self._operand_binding_capture = capture
        self._operand_binding_header = {
            "rank": get_tensor_model_parallel_rank(), "mode": mode, "layers": layers,
            "weight_placeholder_count": len(weights),
            "graph_placeholder_count": len(placeholders),
            "execution_storage_groups": storage_groups,
            "source": "Existing VllmSerializableFunction.optimized_call",
            "timing_source": False,
        }
        return {**self._operand_binding_header, "status": "ARMED"}

    def finish_operand_binding_capture(self):
        """Return the metadata record. Do not run another model call."""
        if not hasattr(self, "_operand_binding_capture"):
            raise RuntimeError("This worker has no operand capture.")
        return {**self._operand_binding_header, **self._operand_binding_capture.finish()}
