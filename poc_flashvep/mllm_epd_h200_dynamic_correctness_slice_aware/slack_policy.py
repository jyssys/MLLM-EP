"""Late-bound selection using frozen causal P-ready and measured D cost models."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from ..mllm_epd_h200_modality_aware_dynamic_split.policy import attention_pairs, cross_image
from .cut_slack import CANDIDATE_BANDS, RATIOS, TOKEN_TARGETS, eligible_geometry


MODES = ("b-new", "c-new-hard", "c-new-soft")
BUDGET_CAPS_MS = (160., 225., 300., 500., None)
CALLBACK_TO_D_SUBMIT_MS = 30.
SUFFIX_BUCKET_ENDS = (512, 2048)


def cost_features(length, prefix):
    suffix = length - prefix
    return length / 8192, suffix / 1024, attention_pairs(prefix, suffix) / (8192 * 1024)


def ready_features(length, prefix, competitors, running, waiting, ready_competitors, batch_size, batch_tokens):
    if None in (competitors, running, waiting, ready_competitors, batch_size, batch_tokens):
        return None
    minimum_issued = min((row["remaining_prompt_tokens"] for row in competitors), default=8192)
    minimum_retired = min((row["prompt_tokens"] - row["retired_prompt_tokens"]
                          for row in competitors), default=8192)
    minimum_pairs = min((attention_pairs(row["issued_prompt_tokens"], row["remaining_prompt_tokens"])
                         for row in competitors), default=8192 * 4096)
    return (
        (length - prefix) / 1024, attention_pairs(prefix, length - prefix) / (8192 * 1024),
        min(8192, minimum_issued) / 1024, min(8192, minimum_retired) / 1024,
        minimum_pairs / (8192 * 1024), min(running, 16) / 4, min(waiting, 32) / 8,
        float(not competitors), float(bool(ready_competitors)),
        min(batch_size, 16) / 4, batch_tokens / 8192,
    )


def predict(model, features):
    return model["intercept"] + sum(
        coefficient * feature for coefficient, feature in zip(model["weights"], features, strict=True))


def suffix_class(suffix):
    return str(next((cap for cap in SUFFIX_BUCKET_ENDS if suffix <= cap), "long"))


def upper_cost(model, features, suffix):
    margin = model.get("upper_margin_by_suffix", {}).get(suffix_class(suffix), model["upper_residual_ms"])
    return max(0., predict(model, features) + margin)


def within_bounds(receipt, length, prefix):
    values = (length, length - prefix, attention_pairs(prefix, length - prefix))
    return all(low <= value <= high for value, (low, high) in zip(values, receipt["bounds"], strict=True))


def cut_offsets(length, ratios=RATIOS):
    remaining = {round(length * ratio) for ratio in ratios if ratio}
    remaining.update(n for n in TOKEN_TARGETS if n <= length * .35)
    return sorted(length - suffix for suffix in remaining if 0 < suffix < length)


@dataclass(frozen=True)
class SlackPolicy:
    mode: str
    cost: dict
    ready: dict
    max_idle_budget_ms: float | None
    safety_ms: float
    natural: bool
    natural_hold: bool
    shadow: bool
    ratios: tuple = RATIOS
    token_cuts: tuple = ()
    gain_estimate: str = "lower"

    def natural_bands(self, length):
        bands = [list(band) for ratio, band in zip(RATIOS[1:], CANDIDATE_BANDS, strict=True) if ratio in self.ratios]
        return bands + [[suffix / length] * 2 for suffix in self.token_cuts if suffix < length]

    def candidate_offsets(self, length):
        bands = self.natural_bands(length)
        cuts = set(cut_offsets(length, self.ratios)) | {length - n for n in self.token_cuts if n < length}
        return [prefix for prefix in sorted(cuts)
                if any(low <= (length - prefix) / length <= high for low, high in bands)]

    def choose(self, *, length, prefix, spans, cuts, competitors, state_observed, E_http_ms=0.,
               progress=None):
        suffix = length - prefix
        span = cross_image(prefix, spans)
        note = {
            "policy": self.mode, "prefix_tokens": prefix, "suffix_tokens": suffix,
            "actual_offload_ratio": suffix / length, "attention_pairs": attention_pairs(prefix, suffix),
            "intra_image_span_tokens": span, "budget_ms": None, "cost_prediction_ms": None,
            "incremental_cost_ms": None, "expected_gain_ms": None,
            "safety_ms": self.safety_ms, "budget_cap_ms": self.max_idle_budget_ms,
        }
        if not eligible_geometry(length, suffix) and suffix not in self.token_cuts:
            return False, "NOT_A_DYNAMIC_GEOMETRY", note
        if not any(low <= suffix / length <= high for low, high in self.natural_bands(length)):
            return False, "OUTSIDE_QUALIFIED_RATIO_BANDS", note
        if self.mode == "c-new-hard" and span:
            return False, "IMAGE_INTERIOR", note
        if not state_observed or not progress:
            return False, "P_PROGRESS_UNKNOWN", note
        if progress["ready_competitors"]:
            return False, "OTHER_P_READY_IN_SAME_STEP", note
        if not within_bounds(self.cost, length, prefix):
            return False, "COST_OUT_OF_SUPPORT", note
        features = ready_features(length, prefix, progress["competitors"], progress["running"],
                                  progress["waiting"], progress["ready_competitors"], progress["batch_size"],
                                  progress["batch_tokens"])
        if features is None:
            return False, "P_PROGRESS_UNKNOWN", note

        p_own = max(0., predict(self.ready["own_ready"], features))
        p_savings = max(0., p_own + self.ready["own_ready"]["lower_residual_ms"])
        next_ready = max(0., predict(self.ready["other_ready"], features))
        causal_budget = max(0., next_ready + self.ready["other_ready"]["lower_residual_ms"])
        budget = causal_budget if self.max_idle_budget_ms is None else min(causal_budget, self.max_idle_budget_ms)
        geometry = cost_features(length, prefix)
        baseline_absolute, baseline_incremental = self.cost["absolute"], self.cost["incremental"]
        modality = self.cost.get("modality_cost", {})
        use_modality = self.mode == "c-new-soft" and modality.get("enabled", False)
        absolute = modality["absolute"] if use_modality else baseline_absolute
        incremental = modality["incremental"] if use_modality else baseline_incremental
        cost_geometry = (*geometry, float(bool(span))) if use_modality else geometry
        d_absolute = upper_cost(absolute, cost_geometry, suffix)
        d_incremental = upper_cost(incremental, cost_geometry, suffix)
        d_incremental_point = max(0., predict(incremental, cost_geometry))
        baseline = upper_cost(baseline_absolute, geometry, suffix)
        handoff = self.cost.get("boundary_to_D_submit_allowance_ms", CALLBACK_TO_D_SUBMIT_MS)
        total = d_absolute + handoff
        lower_gain = p_savings - d_incremental - handoff
        point_gain = p_own - d_incremental_point - handoff
        gain = point_gain if self.gain_estimate == "point" else lower_gain
        note.update(
            cost_prediction_ms=total, incremental_cost_ms=d_incremental,
            incremental_cost_point_ms=d_incremental_point,
            estimated_D_submit_to_first_token_ms=d_absolute,
            boundary_to_D_submit_allowance_ms=handoff,
            P_time_to_other_ready_point_ms=next_ready, P_time_to_own_ready_point_ms=p_own,
            P_continuation_lower_ms=p_savings, budget_ms=budget, uncapped_budget_ms=causal_budget,
            expected_gain_ms=gain, modality_cost_adjustment_ms=d_absolute - baseline,
            modality_cost_enabled=use_modality,
            gain_estimate=self.gain_estimate, lower_gain_ms=lower_gain, point_gain_ms=point_gain,
        )
        if total + self.safety_ms > budget:
            return False, "INSUFFICIENT_PREDICTED_IDLE", note
        if gain <= 0:
            return False, "NO_POSITIVE_EXPECTED_GAIN", note
        return True, "SAFE_FIT_POSITIVE_GAIN", note


def load_policy(mode, cost_path, ready_path, *, max_idle_budget_ms, safety_ms,
                natural, natural_hold, shadow, qualified_ratios=None, qualified_token_cuts=None,
                gain_estimate="lower"):
    if mode not in MODES or not math.isfinite(safety_ms) or safety_ms < 0:
        raise ValueError("Invalid dynamic cut policy")
    if max_idle_budget_ms not in BUDGET_CAPS_MS or natural_hold and not natural or gain_estimate not in ("lower", "point"):
        raise ValueError("Invalid calibrated idle budget or boundary mode")
    if (not shadow and not (qualified_ratios or qualified_token_cuts)
            or any(ratio not in RATIOS[1:] for ratio in qualified_ratios or ())
            or any(suffix not in (*TOKEN_TARGETS, 19) for suffix in qualified_token_cuts or ())):
        raise ValueError("Active serving requires explicitly qualified suffix geometry")
    cost, ready = [json.loads(Path(path).read_text()) for path in (cost_path, ready_path)]
    if cost.get("schema") != "split_cost_v1" or ready.get("schema") != "causal_p_ready_v1":
        raise ValueError("Incorrect model receipt")
    ratios = qualified_ratios or (() if qualified_token_cuts else RATIOS)
    return SlackPolicy(mode, cost, ready, max_idle_budget_ms, safety_ms, natural, natural_hold,
                       shadow, tuple(ratios), tuple(qualified_token_cuts or ()), gain_estimate)
