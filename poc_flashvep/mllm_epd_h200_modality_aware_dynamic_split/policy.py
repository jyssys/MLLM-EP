"""Late decisions using observed state and a common, bounded cut search."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


LATE_KEY = "campaign_late_bound_v1"
POLICIES = ("late10", "generic", "generic_pure", "hard", "soft", "fake_boundary")
RATIOS = (.30, .20, .10, .05)
BLOCK = 128
NATURAL_RATIO_BANDS = {.05: (.025, .075), .10: (.075, .15), .20: (.15, .25), .30: (.25, .35)}


def attention_pairs(prefix, suffix):
    return prefix * suffix + suffix * (suffix + 1) // 2


def cross_image(cut, spans):
    return next((end - start for start, end in spans if start < cut < end), 0)


def candidate_offsets(length, spans, mode, ratios=RATIOS):
    targets = (0.10,) if mode == "late10" else ratios
    ideal = {max(1, min(length - 1, round(length * (1 - ratio)))) for ratio in targets}
    if mode == "late10":
        return sorted(ideal)
    if mode == "generic_pure":
        alternatives = {round(target / BLOCK) * BLOCK for target in ideal}
        return sorted(ideal | {cut for cut in alternatives if 0 < cut < length})
    edges = {edge for span in spans for edge in span if 0 < edge < length}
    for target in tuple(ideal):
        nearby = [edge for edge in edges if abs(edge - target) <= 2 * BLOCK]
        if nearby:
            ideal.add(min(nearby, key=lambda edge: (abs(edge - target), edge)))
    return sorted(ideal)


def equivalent_fake_cuts(cuts, spans, length):
    permitted = sum(not cross_image(cut, spans) for cut in cuts)
    order = sorted(cuts, key=lambda cut: hashlib.sha256(f"{length}:{cut}:placebo".encode()).digest())
    return order[:permitted]


def consume_budget(competitors, peak_prefill_tokens_per_second, limit_ms):
    """Heuristic lower arrival lead from currently known P work; future arrivals are unavailable."""
    leads = [max(0, row["remaining_prompt_tokens"]) * 1000 / peak_prefill_tokens_per_second
             for row in competitors]
    return min([limit_ms, *leads])


@dataclass(frozen=True)
class DecisionPolicy:
    mode: str
    predict_cost: object = None
    peak_prefill_tokens_per_second: float = 80_000
    max_idle_budget_ms: float = 160.0
    safety_ms: float = 10.0
    modality_cost: object = None
    ratios: tuple = RATIOS
    natural: bool = False
    natural_hold: bool = False
    shadow: bool = False
    restrict_late10: bool = False

    def natural_bands(self):
        if self.mode == "late10":
            return [NATURAL_RATIO_BANDS[.10] if self.restrict_late10 else (.05, .15)]
        return [NATURAL_RATIO_BANDS[ratio] for ratio in sorted(self.ratios)]

    def choose(self, *, length, prefix, spans, cuts, competitors, state_observed, E_http_ms=0.0):
        suffix = length - prefix
        boundary_span = cross_image(prefix, spans)
        note = {"policy": self.mode, "prefix_tokens": prefix, "suffix_tokens": suffix,
                "actual_offload_ratio": suffix / length, "attention_pairs": attention_pairs(prefix, suffix),
                "intra_image_span_tokens": boundary_span, "cost_prediction_ms": None,
                "budget_ms": None, "safety_ms": self.safety_ms,
                "predictor": "modality" if self.modality_cost is not None else "generic"}
        if self.mode == "late10":
            return True, "LATE_IDLE_FIXED", note
        if not state_observed:
            return False, "P_PROGRESS_UNKNOWN", note
        if self.mode == "hard" and boundary_span:
            return False, "IMAGE_INTERIOR", note
        if self.mode == "fake_boundary" and prefix not in equivalent_fake_cuts(cuts, spans, length):
            return False, "PLACEBO_ALIGNMENT", note
        budget = consume_budget(competitors, self.peak_prefill_tokens_per_second, self.max_idle_budget_ms)
        note["budget_ms"] = budget
        prediction = self.predict_cost(length, prefix) if self.predict_cost is not None else None
        if prediction is not None and self.modality_cost is not None:
            prediction = self.modality_cost(length, prefix, len(spans), E_http_ms, boundary_span)
        if prediction is None or not math.isfinite(prediction):
            return False, "COST_OUT_OF_SUPPORT", note
        note["cost_prediction_ms"] = prediction
        fits = prediction + self.safety_ms <= budget
        return fits, "SAFE_FIT" if fits else "INSUFFICIENT_PREDICTED_IDLE", note


def load_policy(mode, cost_model=None, *, ratios=RATIOS, max_idle_budget_ms=160.0, safety_ms=10.0,
                natural=False, natural_hold=False, shadow=False, qualified_ratios=None):
    if mode not in POLICIES or not ratios or any(ratio not in RATIOS for ratio in ratios):
        raise ValueError("Invalid late decision policy or ratio set")
    if natural and mode in ("generic_pure", "fake_boundary"):
        raise ValueError("Natural scheduler chunks already share a modality-independent candidate set")
    if natural_hold and not natural:
        raise ValueError("A retirement hold requires a natural P boundary")
    if mode == "late10":
        if qualified_ratios and (tuple(qualified_ratios) != (.10,) or not natural):
            raise ValueError("Scoped Late10 requires the natural 10% ratio band")
        return DecisionPolicy(mode, natural=natural, natural_hold=natural_hold, shadow=shadow,
                              restrict_late10=bool(qualified_ratios))
    if qualified_ratios:
        raise ValueError("Use the causal policy to scope dynamic ratios")
    from .calibration import model_features, suffix_upper_ms

    if cost_model is None:
        raise ValueError("Dynamic ratio selection requires its measured isolated H200 cost model")
    model = json.loads(Path(cost_model).read_text())
    generic = model["generic"]
    bounds = model["training_observed_bounds"]

    def cost(length, prefix):
        values = {"input_tokens": length, "prefix_tokens": prefix, "suffix_tokens": length - prefix}
        if all(bounds[key][0] <= value <= bounds[key][1] for key, value in values.items()):
            return suffix_upper_ms(generic, prefix, length)

    modality = model["modality"]
    enabled = (mode == "soft" and modality["validation_MAE_ms"] < generic["validation_MAE_ms"]
               and modality["validation_p95_underprediction_ms"] <= generic["validation_p95_underprediction_ms"])

    def modality_cost(length, prefix, images, E_http_ms, span):
        suffix = length - prefix
        row = {"input_tokens": length, "prefix_tokens": prefix, "suffix_tokens": suffix,
               "causal_attention_pairs_per_layer": attention_pairs(prefix, suffix),
               "image_count": images, "E_request_http_ms": E_http_ms,
               "cross_boundary_image_count": int(span > 0),
               "cross_boundary_tensor_payload_bytes": span * 3584 * 2}
        center = modality["intercept"] + sum(weight * value for weight, value in
            zip(modality["weights"], model_features(row, "modality"), strict=True))
        return max(0., center) + modality["training_p95_underprediction_ms"]

    return DecisionPolicy(mode, cost, max_idle_budget_ms=max_idle_budget_ms, safety_ms=safety_ms,
                          modality_cost=modality_cost if enabled else None, ratios=ratios,
                          natural=natural, natural_hold=natural_hold, shadow=shadow)
