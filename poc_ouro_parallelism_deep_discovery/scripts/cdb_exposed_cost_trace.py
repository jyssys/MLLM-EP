"""Four-request baseline diagnosis. Frozen 128-token cap, not quality evaluation."""
import argparse
import datetime
import hashlib
import signal
import time
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports, load_engine
from poc_ouro_parallelism_deep_discovery.cdb_cost_observer import (
    CostObserver, EventPool, ScopedMethods, measured_phase_errors, parity_errors,
    trace_coverage_errors, validate_triplet)


def source_checks(contract):
    for relative, expected in contract["source_sha256"].items():
        path = ROOT / relative if relative in ("runtime.py", "scripts/quality_pilot.py") else CDB / "src/looped_cdb" / relative
        assert sha(path) == expected, str(path)
    assert sha(ROOT / contract["selection"]["source"]) == contract["selection"]["source_sha256"]
    rows = read(ROOT / "artifacts/probe_protocol.json")["rows"]
    selected = []
    for dataset in ("gsm8k", "math500"):
        selected.extend(sorted((r for r in rows if r["campaign_partition"] == "calibration" and r["dataset"] == dataset),
                               key=lambda r: r["id"])[:2])
    assert [r["id"] for r in selected] == [r["id"] for r in contract["selection"]["requests"]]
    for row, frozen in zip(selected, contract["selection"]["requests"]):
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == frozen["prompt_utf8_sha256"]
        assert row.get("stop_strings", []) == frozen["stop_strings"]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=["fixed", "adaptive"], required=True)
    args = parser.parse_args()
    contract_path = ROOT / "artifacts/cdb_exposed_cost_trace_contract_v1.json"
    contract = read(contract_path)
    selected = source_checks(contract)
    mask = validate_mask()
    assert len(mask) == 1
    output = Path(os.environ["LOOPED_KV_OUTPUT"]) / "cost_trace"
    output.mkdir(exist_ok=False)
    write(output / "execution_plan.json", dict(contract_sha256=sha(contract_path),
          implementation_sha256={name: sha(ROOT / name) for name in (
              "cdb_cost_observer.py", "scripts/cdb_exposed_cost_trace.py",
              "tests/test_cdb_cost_observer_cpu.py")},
          physical_gpu=mask[0], logical_cuda_rank=0, size=args.size, mode=args.mode,
          scope="Outer baseline diagnosis only. No candidate mechanism or quality result.",
          request_ids=[r["id"] for r in selected], cap_per_generation=128,
          measured_phases=["CLEAN_BEFORE", "TRACE", "CLEAN_AFTER"]))
    configure_imports()
    import torch
    from poc_decode_first_depth_aware_dp.runtime import stats_dict
    model, tokenizer, engine = load_engine(args.size, args.mode, graphs=True, batch=1,
                                          blocks=640, max_model_len=10240, refill=False)
    stream = engine.runner.inputs_and_outputs.compute_stream
    pool = EventPool(torch, stream)
    write(output / "device.json", dict(physical_gpu=mask[0], logical_cuda_rank=0,
          uuid=str(torch.cuda.get_device_properties(0).uuid), model=MODELS[args.size]["id"],
          revision=MODELS[args.size]["revision"], source_contract_sha256=sha(contract_path)))
    depths = []

    def exit_recorder(original):
        def recorded(item, depth):
            depths.append(dict(position=item.token_position, depth=depth + 1))
            return original(item, depth)
        return recorded

    def watchdog(signum, frame):
        raise TimeoutError("Bounded baseline generation exceeded 600 seconds")
    signal.signal(signal.SIGALRM, watchdog)

    def generation(row, phase):
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        stops = [tokenizer.encode(s, add_special_tokens=False) for s in row.get("stop_strings", [])]
        depths.clear()
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        signal.alarm(600)
        try:
            value = engine.generate_batch([ids], max_new_tokens=128, eos_token_id=tokenizer.eos_token_id,
                                          stop_sequences=stops, warmup=False)[0]
            torch.cuda.synchronize()
        finally:
            signal.alarm(0)
        end = time.perf_counter_ns()
        tokens = list(value.generated_tokens)
        assert len(depths) == len(tokens) - 1
        assert all(d["depth"] in ([4] if args.mode == "fixed" else [2, 3, 4]) for d in depths)
        reason = "eos" if tokens and tokens[-1] == tokenizer.eos_token_id else "length" if len(tokens) == 128 else "stop"
        return dict(phase=phase, id=row["id"], dataset=row["dataset"], prompt_ids=ids,
                    output_ids=tokens, generated_tokens=len(tokens), finish_reason=reason,
                    depth_events=list(depths), stats=stats_dict(engine.last_stats),
                    wall_ms=(end - begin) / 1e6, host_start_ns=begin, host_end_ns=end,
                    cap=128, quality_evaluation=False, EOS=tokenizer.eos_token_id)

    with torch.inference_mode(), ScopedMethods() as common:
        common.patch(engine, "_record_exit", exit_recorder)
        summaries = []
        try:
            for row in selected:
                destination = output / row["id"].replace("/", "_")
                destination.mkdir(exist_ok=False)
                phases = {}
                for phase in ("UNTIMED_SAME_REQUEST_WARM", "CLEAN_BEFORE", "TRACE", "CLEAN_AFTER"):
                    if phase == "TRACE":
                        observer = CostObserver(engine, pool)
                        resolved = False
                        try:
                            with observer:
                                result = generation(row, phase)
                            pool.resolve(observer.rows)
                            resolved = True
                            result["event_counts"] = dict(pool.used)
                        finally:
                            # Preserve partial observations without querying CUDA after a failure.
                            write(destination / "trace_intervals.json", observer.rows)
                            write(destination / "trace_export_status.json", dict(
                                elapsed_times_resolved=resolved, event_counts=dict(pool.used),
                                observer_overrides_restored=not observer.saved))
                    else:
                        result = generation(row, phase)
                    phases[phase] = result
                    write(destination / (phase.lower() + ".json"), result)
                    if phase != "UNTIMED_SAME_REQUEST_WARM":
                        phase_errors = measured_phase_errors(result)
                        phase_errors.extend(parity_errors(result, phases["CLEAN_BEFORE"]))
                        if phase == "TRACE":
                            phase_errors.extend(trace_coverage_errors(result))
                        if phase_errors:
                            raise RuntimeError(phase + ": " + str(phase_errors))
                check = validate_triplet(phases)
                write(destination / "validity.json", check)
                summaries.append(dict(id=row["id"], **check))
                if check["errors"]:
                    raise RuntimeError("Attribution rejected: " + str(check["errors"]))
                print("BASELINE_TRACE", args.size, args.mode, row["id"], check["validity"], flush=True)
            write(output / "status.json", dict(success=True, results=summaries,
                  timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  interpretation="Nested outer observations only. No additive E2E attribution or speedup."))
        except BaseException as exc:
            write(output / "failure.json", dict(type=type(exc).__name__, error=str(exc),
                  completed=summaries, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
