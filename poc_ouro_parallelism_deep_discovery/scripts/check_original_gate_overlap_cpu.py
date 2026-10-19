"""Check synthetic gate precision and prefix ordering. All tensors use CPU."""

import itertools
import json

import torch


def serial_policy(hazards):
    cdf = 0.0
    survival = 1.0
    for r, hazard in enumerate(hazards):
        if r == 3:
            return 4
        cdf += hazard * survival
        survival *= 1.0 - hazard
        if r + 1 >= 2 and cdf >= 0.5:
            return r + 1


def mandatory_prefix_policy(hazards):
    pending = tuple(hazards[:2])
    cdf = 0.0
    survival = 1.0
    for hazard in pending:
        cdf += hazard * survival
        survival *= 1.0 - hazard
    if cdf >= 0.5:
        return 2
    hazard = hazards[2]
    cdf += hazard * survival
    survival *= 1.0 - hazard
    return 3 if cdf >= 0.5 else 4


def main():
    x = torch.arange(-512, 513, dtype=torch.float32, device="cpu").div(128).to(torch.bfloat16).unique()
    baseline = torch.sigmoid(x).float()
    promoted = torch.sigmoid(x.float())
    example = None
    for i in range(x.numel()):
        c_baseline = baseline[i].double() + baseline.double() * (1 - baseline[i].double())
        c_promoted = promoted[i].double() + promoted.double() * (1 - promoted[i].double())
        different = (c_baseline >= 0.5) != (c_promoted >= 0.5)
        if different.any():
            j = int(different.nonzero()[0].item())
            example = {
                "logits_BF16": [float(x[i]), float(x[j])],
                "baseline_BF16_sigmoid_hazards": [float(baseline[i]), float(baseline[j])],
                "transport_FP32_before_sigmoid_hazards": [float(promoted[i]), float(promoted[j])],
                "baseline_CDF_R2": float(c_baseline[j]),
                "promoted_CDF_R2": float(c_promoted[j]),
                "baseline_exit_R2": bool(c_baseline[j] >= 0.5),
                "promoted_exit_R2": bool(c_promoted[j] >= 0.5),
            }
            break
    assert example is not None
    cases = 0
    for first_three in itertools.product((0.0, 1 / 16, 1 / 4, 1 / 2, 3 / 4, 15 / 16, 1.0), repeat=3):
        hazards = (*first_three, 0.0)
        assert serial_policy(hazards) == mandatory_prefix_policy(hazards)
        cases += 1
    mutable = [0.25]
    alias = mutable
    snapshot = tuple(mutable)
    mutable[0] = 0.75
    assert alias[0] != snapshot[0]
    print(json.dumps({
        "test": "CPU_GATE_PRECISION_AND_PREFIX_ORDER",
        "torch_version": torch.__version__,
        "GPU_execution": False,
        "hazard_path_counterexample": example,
        "prefix_policy_cases": cases,
        "prefix_policy_equivalence": "PASS",
        "mutable_output_alias_counterexample": "PASS",
        "real_model_outputs": "NOT_RUN",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
