"""One guarded source-template argument diagnostic. Never use this lane for timing."""
import argparse
import copy
import os
from pathlib import Path
import traceback

from poc_ouro_parallelism_deep_discovery.contract import ROOT, GPUS, UUIDS, MODELS, read, sha, validate_mask, write
from poc_ouro_parallelism_deep_discovery.cuda_graph_argument_getter import OPT_IN, validate_policy
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=MODELS)
    parser.add_argument("--tp", required=True, type=int, choices=[1, 2, 4])
    parser.add_argument("--arm", required=True, choices=["native", "alias", "copies"])
    parser.add_argument("--kernel-policy", type=Path,
                        help="Exact source-hashed direct ABI policy. Omission means no argument value reads.")
    return parser.parse_args(argv)


def scoped_config(source, output):
    config = copy.deepcopy(source)
    config["compilation_config"]["cache_dir"] = str(output / "compile_cache")
    config["worker_extension_cls"] = (
        "poc_ouro_parallelism_deep_discovery.graph_argument_probe.ArgumentGraphDebugProbe")
    return config


def frozen_request(workload):
    row = workload["inputs"][0]
    prompt, teacher = list(row["prompt_ids"]["1024"]), list(row["teacher_ids"][:4])
    if len(prompt) != 1024 or len(teacher) != 4 or any(type(t) is not int for t in prompt + teacher):
        raise ValueError("The reference workload lacks the required four-token frozen request.")
    return dict(request_id=row["id"], prompt_token_ids=prompt, teacher_token_ids=teacher)


def reference_for(args):
    tag = {"1.4B": "base14", "2.6B": "base26"}[args.size]
    if args.arm == "native":
        name = f"control_async_{tag}_P{args.tp}_v1"
        return name, ROOT / "outputs" / name / f"{args.size}_R4_TP{args.tp}", None
    if args.tp not in (2, 4):
        raise ValueError("The v4 storage-control references support TP2 or TP4 only.")
    name = f"storage_{args.arm}_{tag}_P{args.tp}_fully_isolated_v4"
    manifest = ROOT / "analysis/storage_provenance_v4" / name / "manifest.json"
    evidence = read(manifest)
    if (evidence["status"] != "PASS" or evidence["size"] != args.size
            or evidence["mode"] != args.arm or evidence["P"] != args.tp):
        raise ValueError("The v4 reference arm lacks matching completed provenance.")
    return name, ROOT / "outputs" / name / f"{args.size}_R4_TP{args.tp}", manifest


def main(argv=None):
    args = parse_args(argv)
    if os.environ.get(OPT_IN) != "1":
        raise RuntimeError(f"Set {OPT_IN}=1 only after review of this diagnostic.")
    policy = validate_policy(read(args.kernel_policy) if args.kernel_policy else {"schema_version": 1, "kernels": {}})
    guard = Path(os.environ["LOOPED_KV_OUTPUT"]).resolve()
    if not guard.is_relative_to(ROOT / "outputs") or not (guard / "run_contract.json").is_file():
        raise RuntimeError("Use the existing campaign gpu_session guard.")
    output = guard / f"{args.size}_R4_TP{args.tp}"
    output.mkdir(exist_ok=False)
    status = dict(success=False, phase="SETUP", evidence_only=True,
                  timing="NOT_MEASURED", natural_quality="NOT_RUN", candidate_speedup="NOT_RUN",
                  executed_operand_identity="UNRESOLVED")
    write(output / "status.json", status)
    try:
        mask = validate_mask()
        if len(mask) != args.tp:
            raise ValueError("The guarded GPU count differs from TP.")
        name, source, manifest = reference_for(args)
        if not read(source / "status.json")["success"]:
            raise ValueError("The reference job did not complete.")
        original = read(source / "engine_config.json")
        if (original["tensor_parallel_size"] != args.tp or original["hf_overrides"]["total_ut_steps"] != 4
                or not original["async_scheduling"] or original["enforce_eager"]
                or original["dtype"] != "bfloat16"
                or Path(original["model"]).resolve() != MODELS[args.size]["path"].resolve()):
            raise ValueError("The reference is not the pinned native-async BF16 R4 graph configuration.")
        inherited = os.environ.get("OURO_CAMPAIGN_STORAGE_CONTROL")
        expected = None if args.arm == "native" else args.arm
        if inherited not in (None, expected):
            raise ValueError("The inherited storage-control environment conflicts with the selected arm.")
        if expected:
            os.environ["OURO_CAMPAIGN_STORAGE_CONTROL"] = expected
        request = frozen_request(read(source / "workload.json"))
        config = scoped_config(original, output)
        roots = {key: str(output / suffix) for key, suffix in [
            ("VLLM_CACHE_ROOT", "vllm_cache"), ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache"),
            ("TRITON_CACHE_DIR", "triton_cache")]}
        for key, path in roots.items():
            if Path(path).exists():
                raise FileExistsError(f"The new diagnostic cache already exists: {path}")
            os.environ[key] = path
        os.environ["LOOPED_KV_OUTPUT"] = str(output)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        write(output / "engine_config.json", config)
        write(output / "kernel_policy.json", policy)
        write(output / "frozen_request.json", request)
        write(output / "reference.json", dict(job=name, path=str(source), arm=args.arm,
            engine_config_sha256=sha(source / "engine_config.json"), workload_sha256=sha(source / "workload.json"),
            manifest=str(manifest) if manifest else None, manifest_sha256=sha(manifest) if manifest else None,
            runner_sha256=sha(Path(__file__)), getter_sha256=sha(ROOT / "cuda_graph_argument_getter.py"),
            worker_sha256=sha(ROOT / "graph_argument_probe.py"), caches=roots,
            policy_path=str(args.kernel_policy.resolve()) if args.kernel_policy else None,
            policy_sha256=sha(args.kernel_policy) if args.kernel_policy else None,
            graph_access="Only one completed request is observed. Export uses the serialized worker RPC.",
            scope="Diagnostic retention changes capture lifetime. No timing or natural-quality claim."))
        configure_native_imports()
        from vllm import LLM, SamplingParams

        status["phase"] = "ENGINE_LOAD"
        write(output / "status.json", status)
        llm = LLM(**config)
        ownership = llm.collective_rpc("inspect_ownership", timeout=120)
        write(output / "ownership.json", ownership)
        if len(ownership) != args.tp or {r["GPU_uuid"].removeprefix("GPU-") for r in ownership} != {
                UUIDS[GPUS.index(g)] for g in mask}:
            raise RuntimeError("The physical GPU ownership differs from the guarded mapping.")
        if any(r["cache_slots"] != 4 * MODELS[args.size]["layers"] for r in ownership):
            raise RuntimeError("The model does not retain all logical-depth KV slots.")
        if args.arm != "native":
            write(output / "storage_control.json", llm.collective_rpc("storage_control_evidence", timeout=120))

        def generate():
            llm.collective_rpc("teacher", timeout=120, args=(request["teacher_token_ids"],))
            result = llm.generate([{"prompt_token_ids": request["prompt_token_ids"]}],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False)
            ids = list(result[0].outputs[0].token_ids)
            if ids != request["teacher_token_ids"]:
                raise RuntimeError("The diagnostic did not return the four frozen teacher IDs.")
            return ids

        for _ in range(3):
            generate()
        llm.collective_rpc("graph_debug_begin", timeout=120)
        status["phase"] = "ONE_OBSERVED_REQUEST"
        write(output / "status.json", status)
        ids = generate()
        graphs = llm.collective_rpc("graph_argument_export", timeout=120,
                                    args=(MODELS[args.size]["layers"], policy))
        if len(graphs) != args.tp or any(r["total_observed_replays"] != 3 for r in graphs):
            raise RuntimeError("The observation does not contain the three expected cached-decode replays per rank.")
        write(output / "argument_observations.json", dict(model=MODELS[args.size]["id"],
            revision=MODELS[args.size]["revision"], R=4, P=args.tp, arm=args.arm,
            physical_gpus=mask, graphs=graphs, output_ids=ids, context=1024, generated_tokens=4,
            request_id=request["request_id"], timing="NOT_MEASURED", natural_quality="NOT_RUN",
            executed_operand_identity="UNRESOLVED", evidence_class="SOURCE_TEMPLATE_ARGUMENT_OBSERVATION"))
        status.update(success=True, phase="SOURCE_TEMPLATE_OBSERVATIONS_EXPORTED", ranks=args.tp)
        write(output / "status.json", status)
    except BaseException as error:
        write(output / "failure.json", dict(error_type=type(error).__name__, phase=status["phase"],
                                              traceback=traceback.format_exc()))
        status.update(success=False, error_type=type(error).__name__)
        write(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
