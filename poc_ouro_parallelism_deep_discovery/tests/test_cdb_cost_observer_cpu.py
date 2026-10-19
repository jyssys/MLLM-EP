import copy
from types import SimpleNamespace as NS
import unittest
from poc_ouro_parallelism_deep_discovery.cdb_cost_observer import (
    CostObserver, EventPool, PARITY_STATS, ScopedMethods, validate_triplet)


class Stream:
    def __init__(self):
        self.syncs = 0
    def synchronize(self):
        self.syncs += 1


class Event:
    def __init__(self, enable_timing):
        assert enable_timing
        self.streams = []
    def record(self, stream):
        self.streams.append(stream)
    def query(self):
        return True
    def elapsed_time(self, end):
        assert self.streams[-1] is end.streams[-1]
        return .25


class Runner:
    def __init__(self, stream):
        self.inputs_and_outputs = NS(compute_stream=stream)
        self.calls = []
    def compute_recurrent_batch(self, items, recurrent_kwargs=None):
        self.calls.append("compute")
        buffers = NS(hidden_in=NS(size=lambda dim: 1))
        self._prepare_recurrent_stage_buffers(buffers, items, real_batch_size=1, bucket_size=1)
        self._run_recurrent_stage(buffers, recurrent_kwargs=recurrent_kwargs or {})
        return "UNCHANGED_RETURN"
    def _prepare_recurrent_stage_buffers(self, buffers, items, **kwargs):
        self.calls.append("metadata")
    def _run_recurrent_stage(self, buffers, recurrent_kwargs):
        self.calls.append("recurrent")
    def _run_coda_forward(self, items):
        self.calls.append("coda")
        return "CODA"
    def stage_coda_batch(self, items, allow_async=True):
        self.calls.append("stage_coda")
        return self._run_coda_forward(items)
    def consume_coda_result(self, pending):
        self.calls.append("consume")
        return [7]


class Engine:
    def __init__(self, stream):
        self.runner = Runner(stream)
        self.cdb_config = NS(max_recurrent_steps=4)
        self._exit_kv_backlog = []
        self.exit_calls = self.flush_calls = 0
    def _should_exit(self, item, recurrent_step, exit_signals, batch_index):
        self.exit_calls += 1
        return recurrent_step == 3
    def _flush_exit_kv_copies(self):
        self.flush_calls += 1
        self._exit_kv_backlog.clear()


def item():
    return NS(state=NS(request_id="a"), token_position=12, recurrent_step=2, hidden_slot=3)


def valid_phases():
    stats = dict.fromkeys(PARITY_STATS, 1)
    stats.update(recurrent_graph_hits=1, recurrent_graph_captures=0, stage_graph_captures=0,
                 stage_graph_hits=2, prelude_batches={"a": 1}, exit_depth_histogram={"2": 1})
    base = dict(prompt_ids=[1], output_ids=[2, 3], generated_tokens=2, finish_reason="length",
                depth_events=[dict(position=1, depth=3)], stats=stats, wall_ms=10)
    phases = {name: copy.deepcopy(base) for name in ("CLEAN_BEFORE", "TRACE", "CLEAN_AFTER")}
    phases["TRACE"]["event_counts"] = dict(recurrent_stage=1, coda=1, publication=1)
    return phases


class ObserverTests(unittest.TestCase):
    def setup_observer(self):
        stream = Stream()
        pool = EventPool(NS(cuda=NS(Event=Event)), stream)
        engine = Engine(stream)
        return engine, pool, stream

    def test_class_method_restoration_and_exception(self):
        engine, _, _ = self.setup_observer()
        original_dict = dict(vars(engine))
        with self.assertRaisesRegex(ValueError, "preserved"):
            with ScopedMethods() as scoped:
                scoped.patch(engine, "_should_exit", lambda original: lambda *a: original(*a))
                self.assertIn("_should_exit", vars(engine))
                raise ValueError("preserved")
        self.assertEqual(vars(engine), original_dict)

    def test_existing_instance_override_restored_by_identity(self):
        engine, _, _ = self.setup_observer()
        original = lambda *args: 71
        engine._should_exit = original
        with ScopedMethods() as scoped:
            scoped.patch(engine, "_should_exit", lambda old: lambda *a: old(*a))
        self.assertIs(engine._should_exit, original)

    def test_duplicate_override_rejected(self):
        engine, _, _ = self.setup_observer()
        with ScopedMethods() as scoped:
            scoped.patch(engine, "_should_exit", lambda x: x)
            with self.assertRaises(RuntimeError):
                scoped.patch(engine, "_should_exit", lambda x: x)

    def test_once_only_calls_nesting_metadata_and_no_added_wait(self):
        engine, pool, stream = self.setup_observer()
        entry = item()
        before = dict(vars(engine.runner))
        with CostObserver(engine, pool) as observer:
            self.assertEqual(engine.runner.compute_recurrent_batch([entry], recurrent_kwargs={"kv_slot": 2}),
                             "UNCHANGED_RETURN")
            self.assertFalse(engine._should_exit(entry, 2, None, 0))
            engine._flush_exit_kv_copies()
            engine._exit_kv_backlog.append((entry, 2))
            engine._flush_exit_kv_copies()
            self.assertEqual(engine.runner.stage_coda_batch([entry], allow_async=False), "CODA")
            self.assertEqual(engine.runner.consume_coda_result(NS(items=[entry])), [7])
        self.assertEqual(stream.syncs, 1)  # Initial pool materialization only.
        self.assertEqual(engine.exit_calls, 1)
        self.assertEqual(engine.flush_calls, 2)
        self.assertEqual(engine.runner.calls, ["compute", "metadata", "recurrent", "stage_coda", "coda", "consume"])
        self.assertEqual(set(vars(engine.runner)), set(before))
        by_kind = {row["kind"]: row for row in observer.rows}
        self.assertEqual(by_kind["metadata_prepare_host"]["parent_id"], by_kind["recurrent_batch_host"]["id"])
        self.assertEqual(by_kind["recurrent_stage_device"]["graph_key"], [2, 1, 1, 2])
        self.assertEqual(pool.used, dict(recurrent_stage=1, publication=1, coda=1))
        entry.token_position = 900
        self.assertEqual(by_kind["recurrent_batch_host"]["items"][0]["position"], 12)
        pool.resolve(observer.rows)
        self.assertEqual(by_kind["coda_device"]["device_span_ms"], .25)

    def test_unknown_stream_rejected(self):
        with self.assertRaises(RuntimeError):
            EventPool(NS(cuda=NS(Event=Event)), None)

    def test_different_stream_rejected(self):
        engine, pool, _ = self.setup_observer()
        engine.runner.inputs_and_outputs.compute_stream = Stream()
        with self.assertRaises(RuntimeError):
            CostObserver(engine, pool)

    def test_pool_overflow_and_unresolved_reuse_rejected(self):
        _, pool, _ = self.setup_observer()
        pool.reset()
        with self.assertRaises(RuntimeError):
            pool.reset()
        for _ in range(128):
            pool.take("publication")
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            pool.take("publication")
        pool.resolve([])
        pool.reset()
        self.assertEqual(pool.used["publication"], 0)

    def test_pair_events_are_unique(self):
        _, pool, _ = self.setup_observer()
        events = [event for pairs in pool.pairs.values() for pair in pairs for event in pair]
        self.assertEqual(len(events), 1536)
        self.assertEqual(len({id(event) for event in events}), 1536)

    def test_nested_overflow_preserves_exception_and_clears_stack(self):
        engine, pool, _ = self.setup_observer()
        observer = CostObserver(engine, pool)
        with self.assertRaisesRegex(RuntimeError, "overflow: recurrent_stage"):
            with observer:
                pool.used["recurrent_stage"] = 512
                engine.runner.compute_recurrent_batch([item()])
        self.assertEqual(observer.stack, [])
        self.assertFalse(observer.saved)
        self.assertEqual(observer.rows[-1]["exception_type"], "RuntimeError")

    def test_event_start_failure_preserves_exception_and_clears_stack(self):
        engine, pool, _ = self.setup_observer()
        def fail(stream):
            raise ValueError("record failed")
        pool.pairs["recurrent_stage"][0][0].record = fail
        observer = CostObserver(engine, pool)
        with self.assertRaisesRegex(ValueError, "record failed"):
            with observer:
                engine.runner.compute_recurrent_batch([item()])
        self.assertEqual(observer.stack, [])
        self.assertFalse(observer.saved)

    def test_valid_triplet(self):
        self.assertEqual(validate_triplet(valid_phases())["validity"], "PASS")

    def test_capture_rejected(self):
        phases = valid_phases()
        phases["TRACE"]["stats"]["stage_graph_captures"] = 1
        self.assertEqual(validate_triplet(phases)["validity"], "ATTRIBUTION_REJECTED")

    def test_eager_stage_execution_rejected(self):
        phases = valid_phases()
        phases["TRACE"]["stats"]["stage_graph_hits"] = 0
        self.assertEqual(validate_triplet(phases)["validity"], "ATTRIBUTION_REJECTED")

    def test_token_depth_stats_finish_parity_rejected(self):
        mutations = [lambda p: p["TRACE"]["output_ids"].append(5),
                     lambda p: p["TRACE"]["depth_events"][0].update(depth=4),
                     lambda p: p["CLEAN_AFTER"]["stats"].update(recurrent_steps=7),
                     lambda p: p["CLEAN_AFTER"].update(finish_reason="eos")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                phases = valid_phases()
                mutate(phases)
                self.assertEqual(validate_triplet(phases)["validity"], "ATTRIBUTION_REJECTED")

    def test_event_and_graph_counts_rejected(self):
        for key in ("coda", "recurrent_stage", "publication"):
            phases = valid_phases()
            phases["TRACE"]["event_counts"][key] = 0
            self.assertFalse(validate_triplet(phases)["attribution_permitted"])
        phases = valid_phases()
        phases["TRACE"]["stats"]["recurrent_graph_hits"] = 0
        self.assertFalse(validate_triplet(phases)["attribution_permitted"])

    def test_fixed_r4_requires_no_publication(self):
        phases = valid_phases()
        for phase in phases.values():
            phase["depth_events"][0]["depth"] = 4
        self.assertFalse(validate_triplet(phases)["attribution_permitted"])
        phases["TRACE"]["event_counts"]["publication"] = 0
        self.assertTrue(validate_triplet(phases)["attribution_permitted"])

    def test_drift_and_overhead_remain_unresolved_not_subtracted(self):
        for phase in ("TRACE", "CLEAN_AFTER"):
            phases = valid_phases()
            phases[phase]["wall_ms"] = 12
            result = validate_triplet(phases)
            self.assertEqual(result["validity"], "OBSERVER_OR_DRIFT_UNRESOLVED")
            self.assertFalse(result["attribution_permitted"])


if __name__ == "__main__":
    unittest.main()
