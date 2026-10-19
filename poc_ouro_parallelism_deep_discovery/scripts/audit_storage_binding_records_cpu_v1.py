"""Recompute checks from saved binding records. Do not import Torch."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["data_ptr", "storage_data_ptr", "storage_offset", "shape", "stride", "dtype"]
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
BLOCK = re.compile(r"L_self_modules_campaign_execution_modules_(\d+)_modules_(\d+)_modules_(.+)_parameters_weight_")
SHARED = {"L_self_modules_embed_tokens_parameters_weight_": "model.embed_tokens.weight",
          "L_self_modules_norm_parameters_weight_": "model.norm.weight"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_row(row, arm):
    failures = []
    def check(condition, code):
        if not condition:
            failures.append(code)
    placeholders = row["placeholder_order"]
    operands = row["operands"]
    snapshot = row["parameter_snapshot"]
    weights = [item for item in operands if "parameters_weight" in item["placeholder"]]
    check([item["index"] for item in operands] == list(range(len(placeholders))), "OPERAND_INDEX_ORDER")
    check([item["placeholder"] for item in operands] == placeholders, "PLACEHOLDER_ORDER")
    check(row["calls_seen"] == 1 and not row["skipped_calls"], "ONE_UNSKIPPED_CALL")
    check(row["original_call_outcome"] == "RETURNED", "ORIGINAL_CALL_RETURNED")
    check(row["context"]["cudagraph_runtime_mode"] == "NONE", "OBSERVED_PREFILL_MODE")
    check("num_tokens=1024" in row["context"]["batch_descriptor"], "OBSERVED_1024_TOKENS")
    check(row["layers"] == 24 and row["mode"] == arm, "MODEL_AND_ARM")
    check(len(weights) == (194 if arm == "alias" else 770), "WEIGHT_COUNT")
    expected_map, slots, field_mismatches = {}, [], []
    for operand in weights:
        name = operand["placeholder"]
        if name in SHARED:
            expected_name = SHARED[name]
        else:
            match = BLOCK.fullmatch(name)
            check(match is not None, f"WEIGHT_NAME:{name}")
            if match is None:
                continue
            depth, layer, role = match.groups()
            check(role in ROLES, f"WEIGHT_ROLE:{name}")
            if role not in ROLES:
                continue
            slots.append((int(depth), int(layer), role))
            expected_name = f"model._campaign_execution.{depth}.{layer}.{ROLES[role]}.weight"
        expected_map[name] = expected_name
        check(operand.get("is_tensor") is True, f"REAL_WEIGHT:{name}")
        check(operand["expected_parameter_name"] == expected_name, f"EXPECTED_NAME:{name}")
        check(expected_name in snapshot, f"SNAPSHOT_NAME:{name}")
        if expected_name not in snapshot:
            continue
        expected = snapshot[expected_name]
        for field in FIELDS:
            if operand[field] != expected[field]:
                field_mismatches.append({"placeholder": name, "field": field})
                failures.append(f"METADATA:{name}:{field}")
        storage_names = sorted(key for key, value in snapshot.items()
                               if value["storage_data_ptr"] == operand["storage_data_ptr"])
        check(sorted(operand["parameter_names_on_storage"]) == storage_names,
              f"STORAGE_OWNER_NAMES:{name}")
        check(operand["dtype"] == "torch.bfloat16", f"BF16_WEIGHT:{name}")
        check(operand["data_ptr"] == operand["storage_data_ptr"] + 2 * operand["storage_offset"],
              f"BF16_POINTER_OFFSET:{name}")
    required_slots = {(depth, layer, role) for depth in ([0] if arm == "alias" else range(4))
                      for layer in range(24) for role in ROLES}
    check(set(slots) == required_slots and len(slots) == len(required_slots), "ALL_BLOCK_SLOTS")
    check(row["expected_weights"] == expected_map, "RECORDED_EXPECTED_MAP")
    groups = row["execution_storage_groups"]
    group_slots = []
    for group in groups:
        layer, role = group["layer"], group["role"]
        group_slots.append((layer, role))
        check(role in ROLES, "GROUP_ROLE")
        if role not in ROLES:
            continue
        names = [f"model._campaign_execution.{depth}.{layer}.{ROLES[role]}.weight" for depth in range(4)]
        pointers = [snapshot[name]["storage_data_ptr"] for name in names]
        check(group["parameter_names"] == names, f"GROUP_PARAMETER_NAMES:{layer}:{role}")
        check(group["storage_data_ptr_by_depth"] == pointers, f"GROUP_POINTERS:{layer}:{role}")
        check(len(set(pointers)) == (1 if arm == "alias" else 4), f"GROUP_OWNERSHIP:{layer}:{role}")
    check(set(group_slots) == {(layer, role) for layer in range(24) for role in ROLES}
          and len(group_slots) == 192, "ALL_EXECUTION_STORAGE_GROUPS")
    return {"rank": row["rank"], "status": "PASS" if not failures else "FAIL",
            "weight_operands_checked": len(weights), "field_comparisons": len(weights) * len(FIELDS),
            "field_mismatches": field_mismatches, "execution_storage_groups_checked": len(groups),
            "storage_offset_values": sorted(set(item["storage_offset"] for item in weights)),
            "weight_dtype_counts": dict(Counter(item["dtype"] for item in weights)),
            "failures": failures, "context": row["context"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    records = []
    negative_checks = []
    for arm in ["alias", "copies"]:
        base = ROOT / "outputs" / f"storage_binding_{arm}_base14_P2_v1"
        run = base / "1.4B_R4_TP2"
        path = run / "operand_binding.json"
        rows = json.loads(path.read_text())
        result = [check_row(row, arm) for row in rows]
        if args.selftest:
            for field in FIELDS:
                bad = copy.deepcopy(rows[0])
                weight = next(item for item in bad["operands"] if "expected_parameter_name" in item)
                weight[field] = (weight[field] + [999] if isinstance(weight[field], list)
                                 else "wrong" if isinstance(weight[field], str) else weight[field] + 1)
                verdict = check_row(bad, arm)
                assert any(item["field"] == field for item in verdict["field_mismatches"])
                negative_checks.append({"arm": arm, "mutated_field": field, "detected": True})
        guard = json.loads((base / "guard_status.json").read_text())
        status = json.loads((run / "status.json").read_text())
        generated = json.loads((run / "generated.json").read_text())
        request = json.loads((run / "frozen_request.json").read_text())
        records.append({"arm": arm, "source": str(path), "source_sha256": sha(path),
                        "guard_success": guard["success"] and guard["exit_code"] == 0,
                        "runner_success": status["success"],
                        "teacher_ids_match": generated["output_ids"] == request["teacher_token_ids"]
                        and len(generated["output_ids"]) == 4,
                        "rows": result})
    print(json.dumps({"schema_version": 1, "evidence_class": "CPU recomputation from saved metadata",
                      "cached_decode_pointer_identity": "NOT_ESTABLISHED",
                      "records": records, "negative_cpu_checks": negative_checks}, indent=2))
    return 0 if all(row["status"] == "PASS" for result in records for row in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
