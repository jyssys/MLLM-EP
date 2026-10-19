"""Scoped outer observers. No Torch import, model edit, or graph-body hook."""
from contextlib import AbstractContextManager
import time


class ScopedMethods(AbstractContextManager):
    """Restore the exact prior instance dictionary, including missing overrides."""

    def __init__(self):
        self.saved = []

    def patch(self, obj, name, wrapper):
        if any(old_obj is obj and old_name == name for old_obj, old_name, *_ in self.saved):
            raise RuntimeError("Duplicate observer target")
        existed = name in vars(obj)
        raw = vars(obj).get(name)
        original = getattr(obj, name)
        self.saved.append((obj, name, existed, raw))
        setattr(obj, name, wrapper(original))

    def __exit__(self, *exception):
        for obj, name, existed, raw in reversed(self.saved):
            if existed:
                setattr(obj, name, raw)
                assert vars(obj)[name] is raw
            else:
                delattr(obj, name)
                assert name not in vars(obj)
        self.saved.clear()
        return False


def item_metadata(items):
    return [dict(request_id=item.state.request_id, position=item.token_position,
                 recurrent_step_zero_based=item.recurrent_step,
                 hidden_slot=item.hidden_slot) for item in items]


class EventPool:
    LIMITS = {"recurrent_stage": 512, "publication": 128, "coda": 128}

    def __init__(self, torch, stream):
        if stream is None:
            raise RuntimeError("Unknown compute stream")
        self.stream = stream
        self.pairs = {name: [(torch.cuda.Event(enable_timing=True),
                             torch.cuda.Event(enable_timing=True)) for _ in range(limit)]
                      for name, limit in self.LIMITS.items()}
        # CUDA events materialize lazily. This happens outside every request clock.
        for pairs in self.pairs.values():
            for start, end in pairs:
                start.record(stream)
                end.record(stream)
        stream.synchronize()
        self.used = dict.fromkeys(self.LIMITS, 0)
        self.resolved = True

    def reset(self):
        if not self.resolved:
            raise RuntimeError("Unresolved timing events cannot be reused")
        self.used = dict.fromkeys(self.LIMITS, 0)
        self.resolved = False

    def take(self, kind):
        index = self.used[kind]
        if index >= len(self.pairs[kind]):
            raise RuntimeError("Timing event pool overflow: " + kind)
        self.used[kind] += 1
        return index, self.pairs[kind][index]

    def resolve(self, rows):
        # Caller must finish the original request-boundary device synchronization.
        for row in rows:
            if "event_kind" in row:
                start, end = self.pairs[row["event_kind"]][row["event_index"]]
                if not end.query():
                    raise RuntimeError("Request boundary did not complete an observed event")
                row["device_span_ms"] = start.elapsed_time(end)
        self.resolved = True


class CostObserver(ScopedMethods):
    def __init__(self, engine, pool, clock=time.perf_counter_ns):
        super().__init__()
        self.engine, self.runner, self.pool, self.clock = engine, engine.runner, pool, clock
        if self.runner.inputs_and_outputs.compute_stream is not pool.stream:
            raise RuntimeError("Observer stream does not match the runner stream")
        self.rows, self.stack = [], []

    def wrapper(self, kind, metadata=None, event_kind=None, enabled=None):
        def make(original):
            def observed(*args, **kwargs):
                meta = {} if metadata is None else metadata(*args, **kwargs)
                row = dict(id=len(self.rows), kind=kind,
                           parent_id=self.stack[-1] if self.stack else None, **meta)
                self.rows.append(row)
                self.stack.append(row["id"])
                pair = None
                row["host_start_ns"] = self.clock()
                start_recorded = False
                primary_exception = False
                try:
                    if event_kind is not None and (enabled is None or enabled(*args, **kwargs)):
                        index, pair = self.pool.take(event_kind)
                        row.update(event_kind=event_kind, event_index=index)
                    if pair is not None:
                        pair[0].record(self.pool.stream)
                        start_recorded = True
                    return original(*args, **kwargs)
                except BaseException as exc:
                    primary_exception = True
                    row["exception_type"] = type(exc).__name__
                    raise
                finally:
                    try:
                        if start_recorded:
                            pair[1].record(self.pool.stream)
                    except BaseException as exc:
                        row["event_cleanup_exception_type"] = type(exc).__name__
                        if not primary_exception:
                            raise
                    finally:
                        row["host_end_ns"] = self.clock()
                        row["host_span_ms"] = (row["host_end_ns"] - row["host_start_ns"]) / 1e6
                        assert self.stack.pop() == row["id"]
            return observed
        return make

    def __enter__(self):
        self.pool.reset()
        try:
            def items(items, *args, **kwargs):
                return dict(items=item_metadata(items))

            def stage(buffers, recurrent_kwargs):
                return dict(graph_key=[2, buffers.hidden_in.size(1),
                                       int(bool(recurrent_kwargs.get("use_early_exit_gate", True))),
                                       int(recurrent_kwargs.get("kv_slot", 0))])

            def exit_meta(item, recurrent_step, exit_signals, batch_index):
                return dict(items=item_metadata([item]), recurrent_step_zero_based=recurrent_step,
                            batch_index=batch_index,
                            forced_final=recurrent_step == self.engine.cdb_config.max_recurrent_steps - 1)

            self.patch(self.runner, "compute_recurrent_batch",
                       self.wrapper("recurrent_batch_host", items))
            self.patch(self.runner, "_prepare_recurrent_stage_buffers",
                       self.wrapper("metadata_prepare_host",
                                    lambda buffers, entries, **kw: dict(items=item_metadata(entries), **kw)))
            self.patch(self.runner, "_run_recurrent_stage",
                       self.wrapper("recurrent_stage_device", stage, "recurrent_stage"))
            self.patch(self.engine, "_should_exit", self.wrapper("exit_decision_host", exit_meta))
            self.patch(self.engine, "_flush_exit_kv_copies",
                       self.wrapper("publication_host_and_device",
                                    lambda: dict(backlog=[dict(items=item_metadata([item]),
                                                              exit_step_zero_based=depth)
                                                         for item, depth in self.engine._exit_kv_backlog]),
                                    "publication", lambda: bool(self.engine._exit_kv_backlog)))
            self.patch(self.runner, "_run_coda_forward", self.wrapper("coda_device", items, "coda"))
            self.patch(self.runner, "stage_coda_batch", self.wrapper("coda_stage_host", items))
            self.patch(self.runner, "consume_coda_result",
                       self.wrapper("coda_consume_host", lambda pending: dict(items=item_metadata(pending.items))))
        except BaseException:
            super().__exit__(None, None, None)
            raise
        return self


PARITY_STATS = ("recurrent_steps", "recurrent_batches", "coda_tokens", "coda_batches",
                "prefill_batches", "prefill_requests", "prefill_recurrent_steps", "prefill_coda_tokens",
                "generated_tokens", "fixed_depth_recurrent_steps", "exit_depth_histogram", "prelude_batches")


def measured_phase_errors(row):
    errors = []
    stats = row["stats"]
    if any(value for key, value in stats.items() if "graph_captures" in key):
        errors.append("measured graph capture")
    if stats["recurrent_graph_hits"] != stats["recurrent_batches"]:
        errors.append("recurrent graph-hit count")
    expected_stage_hits = stats["coda_batches"] + sum(stats["prelude_batches"].values())
    if stats["stage_graph_hits"] != expected_stage_hits:
        errors.append("prelude/coda stage graph-hit count")
    return errors


def parity_errors(row, baseline):
    errors = []
    for field in ("prompt_ids", "output_ids", "finish_reason", "depth_events", "generated_tokens"):
        if row[field] != baseline[field]:
            errors.append("parity " + field)
    for field in PARITY_STATS:
        if row["stats"].get(field) != baseline["stats"].get(field):
            errors.append("parity stats." + field)
    return errors


def trace_coverage_errors(trace):
    errors = []
    expected = dict(recurrent_stage=trace["stats"]["recurrent_batches"],
                    coda=trace["stats"]["coda_batches"],
                    publication=sum(d["depth"] < 4 for d in trace["depth_events"]))
    for kind, count in expected.items():
        if trace["event_counts"][kind] != count:
            errors.append(kind + " event count")
    return errors


def validate_triplet(phases):
    errors = []
    before, trace, after = [phases[name] for name in ("CLEAN_BEFORE", "TRACE", "CLEAN_AFTER")]
    for name, row in phases.items():
        if name == "UNTIMED_SAME_REQUEST_WARM":
            continue
        errors.extend(name + ": " + error for error in measured_phase_errors(row))
        errors.extend(name + ": " + error for error in parity_errors(row, before))
    errors.extend("TRACE: " + error for error in trace_coverage_errors(trace))
    mean_clean = (before["wall_ms"] + after["wall_ms"]) / 2
    drift = abs(after["wall_ms"] - before["wall_ms"]) / mean_clean
    overhead = trace["wall_ms"] / mean_clean - 1
    validity = "PASS" if not errors else "ATTRIBUTION_REJECTED"
    if not errors and (drift > .05 or trace["wall_ms"] > 1.05 * max(before["wall_ms"], after["wall_ms"])):
        validity = "OBSERVER_OR_DRIFT_UNRESOLVED"
    return dict(validity=validity, errors=errors, clean_drift_fraction=drift,
                trace_over_clean_mean_minus_one=overhead,
                attribution_permitted=validity == "PASS")
