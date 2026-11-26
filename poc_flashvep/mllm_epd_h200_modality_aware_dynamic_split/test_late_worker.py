"""Regression tests for P chunk ownership and the original KV handoff path."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import late_worker as worker


def request(*, cuts=None):
    return SimpleNamespace(
        request_id="engine-1", client_index=3,
        kv_transfer_params={
            worker.LATE_KEY: {"request_id": "proxy-1", "cuts": cuts or [600, 800]},
            "do_remote_decode": True,
        },
        prompt_token_ids=list(range(1000)), num_prompt_tokens=1000, max_tokens=1,
        has_encoder_inputs=True, resumable=False, status="running",
        num_in_flight_tokens=0, num_computed_tokens=0, num_output_placeholders=0,
        output_token_ids=[], num_preemptions=0, trace_headers=None,
        take_prefill_stats=lambda: None, take_events=lambda: [],
        get_finished_reason=lambda: "length",
    )


class Scheduler:
    def __init__(self, req):
        self.requests = {req.request_id: req}
        self.running = [req]
        self.waiting = []
        self.skipped_waiting = []
        self.kv_cache_manager = SimpleNamespace()
        self.finished_req_ids_dict = {3: set()}
        self.finished_req_ids = set()

    def _try_schedule_encoder_inputs(self, req, computed, proposed, budget,
                                     shift_computed_tokens=0):
        return [], proposed, budget, []

    def update_from_output(self, scheduled, _runner):
        for req_id, count in scheduled.num_scheduled_tokens.items():
            self.requests[req_id].num_in_flight_tokens -= count
        return {}

    def _update_after_schedule(self, scheduled):
        for req_id, count in scheduled.num_scheduled_tokens.items():
            req = self.requests[req_id]
            req.num_computed_tokens += count
            req.num_in_flight_tokens += count

    def _free_request(self, req):
        assert req.status == "length_capped"
        self.finished_req_ids_dict[req.client_index].add(req.request_id)
        self.finished_req_ids.add(req.request_id)
        return {
            "remote_num_tokens": req.num_computed_tokens,
            "remote_block_ids": [[1, 2]], "do_remote_prefill": True,
        }, None


class EngineOutputs:
    def __init__(self):
        self.outputs = []
        self.finished_requests = None


class CutTest(unittest.TestCase):
    def test_budget_does_not_cross_cut_and_waits_for_actual_gpu_completion(self):
        req = request()
        cut = worker.read_cut_state(req)
        self.assertEqual(worker.capped_rows(cut, 512, 384), 88)
        self.assertEqual(worker.capped_rows(cut, 600, 200), 0)
        req.num_computed_tokens = 600
        req.num_in_flight_tokens = 88
        self.assertIsNone(worker.decide_at_cut(req, "running"))
        req.num_in_flight_tokens = 0
        self.assertIs(worker.decide_at_cut(req, "running"), cut)
        self.assertEqual(len(req.prompt_token_ids), 1000)

    def test_invalid_cut_rejected(self):
        with self.assertRaises(ValueError):
            worker.read_cut_state(request(cuts=[600, 500]))
        with self.assertRaises(ValueError):
            worker.read_cut_state(request(cuts=[True]))

    def test_remote_coordinator_url_rejected(self):
        with self.assertRaises(ValueError):
            worker.CoordinatorDecision("https://example.com/campaign/late-decision", "nonce")
        with self.assertRaises(ValueError):
            worker.CoordinatorDecision("http://127.0.0.1:18847/campaign/late-decision", "")

    def test_decision_timeout_closes_connection_and_denies_cut(self):
        decide = worker.CoordinatorDecision("http://127.0.0.1:18847/campaign/late-decision", "nonce")
        connection = Mock()
        connection.getresponse.side_effect = TimeoutError
        with patch.object(worker.http.client, "HTTPConnection", return_value=connection):
            self.assertEqual(decide({"lease": {}}), (False, "TimeoutError"))
        self.assertEqual(connection.request.call_args.args[-1]["x-campaign-late-nonce"], "nonce")
        connection.close.assert_called_once_with()

    def test_competitor_work_picks_nearest_and_clamps_optimistic_progress(self):
        req = request()
        scheduler = Scheduler(req)
        scheduler.running += [SimpleNamespace(num_prompt_tokens=128, num_computed_tokens=256)]
        scheduler.waiting = [
            SimpleNamespace(num_prompt_tokens=i, num_computed_tokens=0)
            for i in range(512, 525)
        ]
        scheduler.skipped_waiting = [SimpleNamespace(num_prompt_tokens=1200, num_computed_tokens=800)]
        self.assertEqual(
            worker.competitor_work(scheduler, req),
            [{"remaining_prompt_tokens": n} for n in [0, 400, 512, 513, 514, 515, 516, 517]],
        )

    def run_cut(self, accepted, *, natural=False, hold=False, run_ahead=False):
        req = request()
        if natural:
            req.kv_transfer_params[worker.LATE_KEY].update(
                cuts=[], natural=True, ratio_bands=[[.4, .4]], hold=hold)
        scheduler = Scheduler(req)
        records = []
        calls = []
        classes = SimpleNamespace(RUNNING="running", FINISHED_LENGTH_CAPPED="length_capped")

        def module(name, **values):
            item = ModuleType(name)
            vars(item).update(values)
            return item

        modules = {
            "vllm.v1.core.sched.scheduler": module("scheduler", Scheduler=Scheduler),
            "vllm.v1.engine": module("engine", EngineCoreOutputs=EngineOutputs,
                                    EngineCoreOutput=SimpleNamespace),
            "vllm.v1.request": module("request", RequestStatus=classes),
            "poc_flashvep.mllm_epd_prefill_heavy_token_split.common":
                module("common", ROOT=Path("/tmp"), append=lambda path, item: records.append(item)),
        }

        def decision(payload):
            calls.append(payload)
            return accepted, "selected" if accepted else "decode_busy"

        with patch.dict("sys.modules", modules), patch.dict("os.environ", {"MLLM_EPD_RUN_NONCE": "nonce"}), \
                patch.object(worker, "CoordinatorDecision", return_value=decision), \
                patch.object(Scheduler, "_try_schedule_encoder_inputs", Scheduler._try_schedule_encoder_inputs), \
                patch.object(Scheduler, "_update_after_schedule", Scheduler._update_after_schedule), \
                patch.object(Scheduler, "update_from_output", Scheduler.update_from_output):
            worker.install("http://127.0.0.1:18847/campaign/late-decision")
            offers = scheduler._try_schedule_encoder_inputs(req, 0, 1000, 8192)
            self.assertEqual(offers[1], 1000 if natural else 600)
            actual = SimpleNamespace(num_scheduled_tokens={req.request_id: 600})
            scheduler._update_after_schedule(actual)
            wait = scheduler._try_schedule_encoder_inputs(req, 600, 400, 8192)
            self.assertEqual(wait[1], 400 if natural and not hold else 0)
            if run_ahead:
                req.num_in_flight_tokens += 250
                req.num_computed_tokens += 250
            outputs = scheduler.update_from_output(actual, None)
            if run_ahead:
                self.assertEqual(calls, [])
                self.assertEqual(outputs, {})
                self.assertEqual(records[0]["later_tokens_inflight"], 250)
                self.assertEqual(records[0]["prefix_tokens"], 600)
                self.assertIs(scheduler.running[0], req)
                return
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["lease"], {"request_id": "proxy-1", "prefix_tokens": 600})
            self.assertEqual(calls[0]["competitors"], [])
            self.assertTrue(calls[0]["competitors_observed"])
            if natural:
                self.assertTrue(calls[0]["natural"])
                self.assertEqual(calls[0]["scheduled_chunk_tokens"], 600)
                self.assertEqual(calls[0]["hold"], hold)
                self.assertEqual(calls[0]["held_schedule_attempts"], int(hold))
            self.assertEqual(len(req.prompt_token_ids), 1000)
            self.assertEqual(req.num_computed_tokens, 600)
            self.assertEqual(len(req.output_token_ids), 0)
            if accepted:
                self.assertNotIn(req, scheduler.running)
                self.assertEqual(outputs[3].outputs[0].new_token_ids, [])
                self.assertEqual(outputs[3].outputs[0].kv_transfer_params[worker.PLAN_KEY],
                                 {"full_prompt_tokens": 1000, "prefix_tokens": 600})
                self.assertEqual(outputs[3].finished_requests, {req.request_id})
            else:
                self.assertIs(scheduler.running[0], req)
                self.assertEqual(outputs, {})
                self.assertEqual(scheduler._try_schedule_encoder_inputs(req, 600, 400, 8192)[1],
                                 400 if natural else 200)
            self.assertEqual(records[0]["accepted"], accepted)

    def test_denial_keeps_original_request_and_schedules_remainder(self):
        self.run_cut(False)

    def test_acceptance_exports_actual_boundary_without_p_sampling(self):
        self.run_cut(True)

    def test_natural_mode_keeps_original_rows_before_and_after_denial(self):
        self.run_cut(False, natural=True)

    def test_natural_mode_can_export_actual_retired_native_boundary(self):
        self.run_cut(True, natural=True)

    def test_natural_mode_skips_when_native_scheduler_already_issued_more_work(self):
        self.run_cut(True, natural=True, run_ahead=True)

    def test_natural_hold_preserves_armed_chunk_and_releases_next_rows_on_denial(self):
        self.run_cut(False, natural=True, hold=True)

    def test_natural_hold_exports_same_natural_rows_after_retirement(self):
        self.run_cut(True, natural=True, hold=True)


if __name__ == "__main__":
    unittest.main()
