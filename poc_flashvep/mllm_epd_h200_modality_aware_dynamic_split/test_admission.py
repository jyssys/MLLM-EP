import asyncio
import time
import unittest

from .admission import LateAdmission
from .coordinator import LateCoordinator, Prefill
from .policy import DecisionPolicy, candidate_offsets, cross_image, equivalent_fake_cuts


class AdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_checks_idle_but_never_reserves_d(self):
        class IdleWorkerState:
            def post(self, *_args, **_kwargs):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            def raise_for_status(self):
                pass

            async def json(self):
                return [{"schedulers": [{"running": 0, "waiting": 0, "KV_free_blocks": 512}]}]

        gate = LateAdmission(8)
        worker = LateCoordinator(gate, DecisionPolicy("late10", shadow=True), "http://127.0.0.1:18842", ".")
        worker.prefills["request"] = Prefill(1000, [900], [], 256, 100.0, [])
        result = await worker.decide({"lease": {"request_id": "request", "prefix_tokens": 900},
                                      "full_prompt_tokens": 1000, "boundary_ready": time.monotonic()},
                                     IdleWorkerState())
        self.assertEqual(result, {"accepted": False, "reason": "SHADOW_ELIGIBLE"})
        self.assertTrue(gate.idle_for_reservation())

    async def test_expired_p_callback_never_holds_d(self):
        gate = LateAdmission(8)
        self.assertIsNone(await gate.reserve("expired", 800, {}, deadline=0.0))
        self.assertTrue(gate.idle_for_reservation())

    async def test_busy_d_never_queues_a_suffix_and_keeps_native_batching(self):
        gate = LateAdmission(8)
        await gate.enter("native_1", False)
        self.assertIsNone(await gate.reserve("partial", 800, {}))
        await gate.enter("native_2", False)
        self.assertEqual(set(gate.active), {"native_1", "native_2"})

    async def test_native_ready_joins_after_first_token_without_waiting_for_eos(self):
        gate = LateAdmission(8)
        await gate.reserve("partial", 800, {})
        pending = asyncio.create_task(gate.enter("native", False))
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        self.assertIsNone(await gate.reserve("other", 800, {}))
        await gate.claim("partial", 800)
        self.assertFalse(pending.done())
        await gate.first_token("partial")
        await asyncio.wait_for(pending, 1)
        self.assertEqual(set(gate.active), {"partial", "native"})

    async def test_native_continuation_releases_only_its_unclaimed_reservation(self):
        gate = LateAdmission(8)
        await gate.reserve("partial", 800, {})
        self.assertFalse(await gate.release_unclaimed("not_owner"))
        with self.assertRaises(RuntimeError):
            await gate.claim("partial", 799)
        pending = asyncio.create_task(gate.enter("native", False))
        self.assertTrue(await gate.release_unclaimed("partial"))
        await asyncio.wait_for(pending, 1)
        self.assertEqual(set(gate.active), {"native"})


class PolicyTest(unittest.TestCase):
    def test_same_candidate_count_placement_and_mode_cost(self):
        spans = [(15, 1400), (1410, 2995)]
        cuts = candidate_offsets(3030, spans, "generic")
        self.assertEqual(cuts, candidate_offsets(3030, spans, "hard"))
        self.assertEqual(cuts, candidate_offsets(3030, spans, "soft"))
        placebo = equivalent_fake_cuts(cuts, spans, 3030)
        self.assertEqual(len(placebo), sum(not cross_image(cut, spans) for cut in cuts))
        features = dict(length=3030, prefix=2727, spans=spans, cuts=cuts,
                        competitors=[], state_observed=True)
        cost = lambda *_: 40.0
        self.assertTrue(DecisionPolicy("generic", cost).choose(**features)[0])
        self.assertFalse(DecisionPolicy("hard", cost).choose(**features)[0])
        self.assertTrue(DecisionPolicy("soft", cost).choose(**features)[0])
        features["competitors"] = [{"remaining_prompt_tokens": 20}]
        self.assertFalse(DecisionPolicy("generic", cost).choose(**features)[0])


if __name__ == "__main__":
    unittest.main()
