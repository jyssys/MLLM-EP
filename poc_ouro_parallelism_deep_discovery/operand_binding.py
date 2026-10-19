"""Opt-in metadata capture at one optimized-call boundary. No Torch import."""

from collections import defaultdict
import copy
import re


FIELDS = ("data_ptr", "storage_data_ptr", "storage_offset", "shape", "stride", "dtype")
ROLES = {
    "input_layernorm": "input_layernorm",
    "input_layernorm_2": "input_layernorm_2",
    "post_attention_layernorm": "post_attention_layernorm",
    "post_attention_layernorm_2": "post_attention_layernorm_2",
    "self_attn_modules_qkv_proj": "self_attn.qkv_proj",
    "self_attn_modules_o_proj": "self_attn.o_proj",
    "mlp_modules_gate_up_proj": "mlp.gate_up_proj",
    "mlp_modules_down_proj": "mlp.down_proj",
}
PATTERN = re.compile(
    r"L_self_modules_campaign_execution_modules_(\d+)_modules_(\d+)_modules_"
    r"(.+)_parameters_weight_", re.IGNORECASE,
)
SHARED = {
    "l_self_modules_embed_tokens_parameters_weight_": "model.embed_tokens.weight",
    "l_self_modules_norm_parameters_weight_": "model.norm.weight",
}


def expected_weight_bindings(placeholders, mode, layers):
    """Check the complete storage-control graph and map its weight names."""
    if mode not in {"alias", "copies"} or layers not in {24, 48}:
        raise ValueError("The model size or storage-control mode is not supported.")
    weights = [name for name in placeholders if "parameters_weight" in name.lower()]
    depths = [0] if mode == "alias" else range(4)
    slots, shared, expected = [], [], {}
    for name in weights:
        if name.lower() in SHARED:
            shared.append(name.lower())
            expected[name] = SHARED[name.lower()]
            continue
        match = PATTERN.fullmatch(name)
        if match is None or match.group(3).lower() not in ROLES:
            raise ValueError("The graph has an unrecognized weight placeholder.")
        depth, layer, role = match.groups()
        role = role.lower()
        slots.append((int(depth), int(layer), role))
        expected[name] = f"model._campaign_execution.{depth}.{layer}.{ROLES[role]}.weight"
    required = {(depth, layer, role) for depth in depths
                for layer in range(layers) for role in ROLES}
    if (set(slots) != required or len(slots) != len(required)
            or set(shared) != set(SHARED) or len(shared) != 2):
        raise ValueError("The graph has incorrect weight roles or storage-depth prefixes.")
    return expected


def tensor_metadata(value):
    """Read tensor metadata. Do not read tensor values."""
    return {
        "data_ptr": value.data_ptr(),
        "storage_data_ptr": value.untyped_storage().data_ptr(),
        "storage_offset": value.storage_offset(),
        "shape": list(value.shape), "stride": list(value.stride()),
        "dtype": str(value.dtype),
    }


def parameter_snapshot(named_parameters):
    """Record all names, including aliases, after parameter loading."""
    return {name: tensor_metadata(value) for name, value in named_parameters}


class LiftedOperandBindingCapture:
    """Wrap one existing callable. Restore it before the selected call."""

    def __init__(self, owner, placeholders, parameters, expected_weights,
                 is_real_tensor, context_provider=lambda: {}):
        if not callable(getattr(owner, "optimized_call", None)):
            raise ValueError("The optimized call does not exist.")
        if getattr(owner, "_campaign_operand_binding_capture", None) is not None:
            raise ValueError("This callable already has an operand capture.")
        if not placeholders or len(set(placeholders)) != len(placeholders):
            raise ValueError("Placeholder names must be present and unique.")
        if not expected_weights or not set(expected_weights) <= set(placeholders):
            raise ValueError("Expected weight placeholders do not match the graph.")
        if not set(expected_weights.values()) <= set(parameters):
            raise ValueError("Expected parameter names do not match the loaded model.")
        self.owner = owner
        self.original = owner.optimized_call
        self.placeholders = list(placeholders)
        self.weight_indices = [index for index, name in enumerate(self.placeholders)
                               if name in expected_weights]
        self.parameters = copy.deepcopy(parameters)
        self.expected_weights = dict(expected_weights)
        self.is_real_tensor = is_real_tensor
        self.context_provider = context_provider
        self.storage_names = defaultdict(list)
        for name, metadata in self.parameters.items():
            self.storage_names[metadata["storage_data_ptr"]].append(name)
        self.record = {"status": "ARMED", "calls_seen": 0, "skipped_calls": [],
                       "placeholder_order": self.placeholders,
                       "expected_weights": self.expected_weights,
                       "parameter_snapshot": self.parameters,
                       "scope": "One lifted-operand binding. Not kernel reads or timing.",
                       "metadata_only": True}

        def observed(*args, **kwargs):
            self.record["calls_seen"] += 1
            reason = None
            if kwargs:
                reason = "KEYWORD_ARGUMENTS_NOT_SUPPORTED"
            elif len(args) != len(self.placeholders):
                reason = "POSITIONAL_ARITY_MISMATCH"
            else:
                try:
                    if not all(self.is_real_tensor(args[index]) for index in self.weight_indices):
                        reason = "WEIGHT_ARGUMENT_IS_NOT_A_REAL_TENSOR"
                except Exception as error:
                    reason = f"TENSOR_CLASSIFICATION_ERROR:{type(error).__name__}"
            if reason is not None:
                if len(self.record["skipped_calls"]) < 8:
                    self.record["skipped_calls"].append(reason)
                return self.original(*args, **kwargs)
            # Restore before execution. Later calls use the unmodified callable.
            self.restore()
            try:
                operands = []
                for index, (name, value) in enumerate(zip(self.placeholders, args)):
                    operand = {"index": index, "placeholder": name,
                               "is_tensor": self.is_real_tensor(value)}
                    if operand["is_tensor"]:
                        metadata = tensor_metadata(value)
                        operand.update(metadata)
                        operand["parameter_names_on_storage"] = self.storage_names.get(
                            metadata["storage_data_ptr"], [])
                    if name in self.expected_weights:
                        expected_name = self.expected_weights[name]
                        expected = self.parameters[expected_name]
                        operand["expected_parameter_name"] = expected_name
                        operand["matches_expected_parameter_metadata"] = all(
                            operand[field] == expected[field] for field in FIELDS)
                    operands.append(operand)
                self.record["operands"] = operands
                self.record["context"] = self.context_provider()
                matches = [item["matches_expected_parameter_metadata"] for item in operands
                           if "matches_expected_parameter_metadata" in item]
                self.record["status"] = "CAPTURED_MATCH" if all(matches) else "CAPTURED_MISMATCH"
            except Exception as error:
                self.record["status"] = "CAPTURE_ERROR"
                self.record["capture_error_type"] = type(error).__name__
            try:
                result = self.original(*args, **kwargs)
            except BaseException as error:
                self.record["original_call_outcome"] = "RAISED"
                self.record["original_error_type"] = type(error).__name__
                raise
            self.record["original_call_outcome"] = "RETURNED"
            return result

        self.wrapper = observed
        owner._campaign_operand_binding_capture = self
        owner.optimized_call = self.wrapper

    def restore(self):
        if self.owner.optimized_call is self.wrapper:
            self.owner.optimized_call = self.original

    def finish(self):
        self.restore()
        if self.record["status"] == "ARMED":
            self.record["status"] = "NOT_CAPTURED"
        return copy.deepcopy(self.record)
