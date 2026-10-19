"""One guarded, evidence-only storage-binding request. This is not timing."""

import argparse
import copy
import os
from pathlib import Path
import traceback

from poc_ouro_parallelism_deep_discovery.contract import ROOT, GPUS, UUIDS, MODELS, read, sha, validate_mask, write
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=["1.4B", "2.6B"])
    parser.add_argument("--tp", required=True, type=int, choices=[2, 4])
    parser.add_argument("--arm", required=True, choices=["alias", "copies"])
    return parser.parse_args(argv)


def scoped_config(source, output):
    config = copy.deepcopy(source)
    config["compilation_config"]["cache_dir"] = str(output / "compile_cache")
    config["worker_extension_cls"] = (
        "poc_ouro_parallelism_deep_discovery.operand_binding_worker.OperandBindingProbe"
    )
    return config


def frozen_request(workload):
    row = workload["inputs"][0]
    prompt = list(row["prompt_ids"]["1024"])
    teacher = list(row["teacher_ids"][:4])
    if len(prompt) != 1024 or len(teacher) != 4:
        raise ValueError("The reference workload does not contain the required frozen request.")
    if not all(isinstance(token, int) for token in prompt + teacher):
        raise ValueError("The frozen request contains a non-integer token ID.")
    return {"request_id": row["id"], "prompt_token_ids": prompt, "teacher_token_ids": teacher}


def main(argv=None):
    args = parse_args(argv)
    guard_root = Path(os.environ["LOOPED_KV_OUTPUT"]).resolve()
    if not guard_root.is_relative_to(ROOT / "outputs") or not (guard_root / "run_contract.json").exists():
        raise RuntimeError("Run this evidence lane through the campaign gpu_session guard.")
    output = guard_root / f"{args.size}_R4_TP{args.tp}"
    output.mkdir(parents=True, exist_ok=False)
    status = {"success": False, "phase": "SETUP", "evidence_only": True,
              "timing_source": False, "quality": "NOT_RUN",
              "scope": "One observed optimized-call binding. Not cached decode kernel parameters."}
    write(output / "status.json", status)
    llm, capture_armed = None, False
    try:
        mask = validate_mask()
        if len(mask) != args.tp:
            raise ValueError("The guarded GPU count does not match TP.")
        model_tag = {"1.4B": "base14", "2.6B": "base26"}[args.size]
        reference_name = f"storage_{args.arm}_{model_tag}_P{args.tp}_fully_isolated_v4"
        reference = ROOT / "outputs" / reference_name / f"{args.size}_R4_TP{args.tp}"
        manifest_path = ROOT / "analysis/storage_provenance_v4" / reference_name / "manifest.json"
        manifest = read(manifest_path)
        if (manifest["status"] != "PASS" or manifest["size"] != args.size
                or manifest["mode"] != args.arm or manifest["P"] != args.tp
                or not read(reference / "status.json")["success"]):
            raise RuntimeError("The v4 reference arm lacks a completed graph-provenance check.")
        source_config = read(reference / "engine_config.json")
        if source_config["tensor_parallel_size"] != args.tp or source_config["hf_overrides"]["total_ut_steps"] != 4:
            raise ValueError("The v4 reference configuration does not match this request.")
        if Path(source_config['model']).resolve() != MODELS[args.size]['path'].resolve():
            raise ValueError('The model revision differs from the pinned campaign checkpoint.')
        request = frozen_request(read(reference / "workload.json"))
        config = scoped_config(source_config, output)
        roots = {name: str(output / suffix) for name, suffix in [
            ("VLLM_CACHE_ROOT", "vllm_cache"),
            ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache"),
            ("TRITON_CACHE_DIR", "triton_cache"),
        ]}
        for name, path in roots.items():
            if Path(path).exists():
                raise FileExistsError(f"The new job cache already exists: {path}")
            os.environ[name] = path
        if Path(config["compilation_config"]["cache_dir"]).exists():
            raise FileExistsError("The new ordinary compile cache already exists.")
        os.environ["OURO_CAMPAIGN_STORAGE_CONTROL"] = args.arm
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        write(output / "cache_environment.json", {
            "before_vllm_import": True, "variables": roots,
            "storage_control": args.arm, "cuda_allocator": "expandable_segments:False",
            "scope": "Evidence-only job roots. AOT can relocate Inductor below VLLM_CACHE_ROOT.",
        })
        write(output / "reference.json", {
            "job": reference_name, "run_path": str(reference),
            "engine_config_sha256": sha(reference / "engine_config.json"),
            "workload_sha256": sha(reference / "workload.json"),
            "provenance_manifest": str(manifest_path), "provenance_manifest_sha256": sha(manifest_path),
            "runner_sha256": sha(Path(__file__)),
            "binding_core_sha256": sha(ROOT / "operand_binding.py"),
            "binding_worker_sha256": sha(ROOT / "operand_binding_worker.py"),
        })
        write(output / "engine_config.json", config)
        write(output / "frozen_request.json", request)
        configure_native_imports()
        from vllm import LLM, SamplingParams

        status["phase"] = "ENGINE_LOAD"
        write(output / "status.json", status)
        llm = LLM(**config)
        ownership = llm.collective_rpc("inspect_ownership", timeout=120)
        write(output / "ownership.json", ownership)
        if len(ownership) != args.tp or {row['GPU_uuid'].removeprefix('GPU-') for row in ownership} != {
                UUIDS[GPUS.index(gpu)] for gpu in mask}:
            raise RuntimeError('The loaded physical GPU ownership differs from the guarded mapping.')
        if not all(row['cache_slots'] == 4 * MODELS[args.size]['layers'] for row in ownership):
            raise RuntimeError('The loaded cache ownership does not retain all depth-indexed slots.')
        write(output / "storage_control.json", llm.collective_rpc("storage_control_evidence", timeout=120))
        llm.collective_rpc("teacher", timeout=120, args=(request["teacher_token_ids"],))
        write(output / "capture_arm.json", llm.collective_rpc("start_operand_binding_capture", timeout=120))
        capture_armed = True
        status["phase"] = "ONE_EVIDENCE_REQUEST"
        write(output / "status.json", status)
        results = llm.generate([{"prompt_token_ids": request["prompt_token_ids"]}],
                               SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
                               use_tqdm=False)
        captures = llm.collective_rpc("finish_operand_binding_capture", timeout=120)
        capture_armed = False
        write(output / "operand_binding.json", captures)
        token_ids = list(results[0].outputs[0].token_ids)
        write(output / "generated.json", {"output_ids": token_ids,
              "expected_teacher_ids": request["teacher_token_ids"], "timing_source": False})
        if token_ids != request["teacher_token_ids"]:
            raise RuntimeError("The evidence request did not return the four frozen teacher IDs.")
        if len(captures) != args.tp or not all(
                row["status"] == "CAPTURED_MATCH" and row.get("original_call_outcome") == "RETURNED"
                for row in captures):
            raise RuntimeError("At least one rank lacks a matching observed optimized-call binding.")
        status.update(success=True, phase="EVIDENCE_CAPTURED", ranks=args.tp)
        write(output / "status.json", status)
    except BaseException as error:
        if capture_armed and llm is not None:
            try:
                write(output / "operand_binding_after_failure.json",
                      llm.collective_rpc("finish_operand_binding_capture", timeout=120))
            except BaseException as finish_error:
                write(output / "capture_finish_failure.json", {
                    "error_type": type(finish_error).__name__, "traceback": traceback.format_exc()})
        write(output / "failure.json", {"error_type": type(error).__name__,
              "phase": status["phase"], "traceback": traceback.format_exc()})
        status.update(success=False, error_type=type(error).__name__)
        write(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
