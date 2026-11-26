from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from ..mllm_epd_h200_modality_aware_dynamic_split.coordinator import LateCoordinator, Prefill
from ..mllm_epd_h200_modality_aware_dynamic_split.policy import load_policy as load_existing_policy
from .cut_slack import _gap, merge_intervals
from .slack_policy import SlackPolicy


def constant(intercept, count, lower=0., upper=0.):
    return {"intercept": intercept, "weights": [0.] * count,
            "lower_residual_ms": lower, "upper_residual_ms": upper}


def candidate(policy, prefix=9000, ready=0):
    return policy.choose(
        length=10000, prefix=prefix, spans=[(200, 9200)], cuts=[], competitors=[],
        state_observed=True, progress={"competitors": [], "running": 1, "waiting": 2,
                                       "ready_competitors": ready, "batch_size": 2, "batch_tokens": 8192})


class SlackTests(TestCase):
    def setUp(self):
        self.policy = SlackPolicy(
            "b-new",
            cost={"bounds": [[2000, 20000], [19, 4096], [1, 100_000_000]],
                  "absolute": constant(120., 3, upper=15.), "incremental": constant(65., 3, upper=8.)},
            ready={"own_ready": constant(270., 11, lower=-20.),
                   "other_ready": constant(300., 11, lower=-30.)},
            max_idle_budget_ms=225., safety_ms=10., natural=True, natural_hold=False, shadow=False,
            ratios=(.10,),
        )

    def test_cut_idle_distinguishes_busy_from_censoring(self):
        merged = merge_intervals([(3., 4.), (5., 6.), (3.5, 4.5)])
        self.assertEqual(_gap(merged, [3., 5.], 4., 5.5), (0., False))
        self.assertEqual(_gap(merged, [3., 5.], 4.5, 5.5), (500., False))
        self.assertEqual(_gap(merged, [3., 5.], 4.5, 4.75), (250., True))

    def test_fit_charges_absolute_cost_and_gain_uses_incremental_cost(self):
        ok, _, row = candidate(self.policy)
        self.assertTrue(ok)
        self.assertEqual(row["cost_prediction_ms"], 165.)
        self.assertEqual(row["expected_gain_ms"], 147.)
        too_short = replace(self.policy, max_idle_budget_ms=160.)
        self.assertEqual(candidate(too_short)[1], "INSUFFICIENT_PREDICTED_IDLE")

    def test_prepared_competitor_and_unqualified_geometry_fall_back(self):
        self.assertEqual(candidate(self.policy, ready=1)[1], "OTHER_P_READY_IN_SAME_STEP")
        self.assertEqual(candidate(self.policy, prefix=9500)[1], "OUTSIDE_QUALIFIED_RATIO_BANDS")
        exact = replace(self.policy, ratios=(), token_cuts=(64,))
        self.assertTrue(candidate(exact, prefix=9936)[0])
        self.assertEqual(candidate(exact, prefix=9935)[1], "OUTSIDE_QUALIFIED_RATIO_BANDS")
        self.assertEqual(candidate(replace(self.policy, mode="c-new-hard"))[1], "IMAGE_INTERIOR")

    def test_slice_transport_policy_without_measured_penalty_is_identical(self):
        generic = candidate(self.policy)[2]
        soft = candidate(replace(self.policy, mode="c-new-soft"))[2]
        self.assertEqual(generic["expected_gain_ms"], soft["expected_gain_ms"])
        self.assertEqual(generic["cost_prediction_ms"], soft["cost_prediction_ms"])

    def test_point_gain_rechecks_the_same_absolute_occupancy_limit(self):
        normal = candidate(self.policy)[2]
        point_policy = replace(self.policy, gain_estimate="point")
        point = candidate(point_policy)[2]
        self.assertEqual(point["cost_prediction_ms"], normal["cost_prediction_ms"])
        self.assertEqual(point["budget_ms"], normal["budget_ms"])
        self.assertGreater(point["expected_gain_ms"], normal["expected_gain_ms"])
        self.assertEqual(candidate(replace(point_policy, max_idle_budget_ms=160.))[1],
                         "INSUFFICIENT_PREDICTED_IDLE")

    def test_transport_guard_requires_a_live_descriptor_for_each_needed_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            coordinator = LateCoordinator(None, None, None, root / "processed_prompts",
                                          require_ready_transport=True)
            item = Prefill(1000, [800], [(200, 850), (900, 950)], 256, 10., [],
                           image_keys=("crossing_image", "suffix_image"))
            crossing, suffix = (root / key for key in item.image_keys)
            for folder in (crossing, suffix):
                folder.mkdir()
                (folder / "cuda_ipc_ready.json").touch()
                (folder / "cuda_ipc_inline_prefill.json").touch()
            (crossing / "cuda_ipc_inline_prefill_taken.json").touch()
            self.assertEqual(coordinator.ready_suffix_transport(item, 800), (False, 2, 1))
            (crossing / "cuda_ipc_inline_decode_crossing.json").touch()
            self.assertEqual(coordinator.ready_suffix_transport(item, 800), (True, 2, 0))
            (crossing / "cuda_ipc_inline_decode_crossing_taken.json").touch()
            self.assertEqual(coordinator.ready_suffix_transport(item, 800), (False, 2, 1))
            self.assertEqual(coordinator.ready_suffix_transport(item, 850), (True, 1, 0))

    def test_late10_scope_is_explicit_and_preserves_original_natural_band(self):
        original = load_existing_policy("late10", natural=True)
        scoped = load_existing_policy("late10", natural=True, qualified_ratios=(.10,))
        self.assertEqual(original.natural_bands(), [(.05, .15)])
        self.assertEqual(scoped.natural_bands(), [(.075, .15)])
