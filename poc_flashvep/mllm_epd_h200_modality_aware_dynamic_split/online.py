"""Run Native and late-bound online cohorts on the verified three-GPU lease."""
import argparse
import asyncio
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import rpc
from ..mllm_epd_h200_wide_decode_batch_token_split.host import snapshot
from ..mllm_epd_h200_wide_decode_batch_token_split.online import (
    SEED, configure, fingerprint_source, measured_cohorts,
)
from ..mllm_epd_h200_dynamic_correctness_slice_aware.slack_policy import MODES as CAUSAL_POLICIES
from .policy import POLICIES


LATE_POLICIES = (*POLICIES, *CAUSAL_POLICIES)
MODES = ("native-strong", "native-common", "native-matched", "idle10", *LATE_POLICIES)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--decode-cap", choices=("8", "free"), default="8")
    parser.add_argument("--qps", type=float, nargs="+", default=[])
    parser.add_argument("--subset", choices=("all", "discovery", "heldout"), default="all")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--paired-start-id")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--observe-batch", action="store_true",
                        help="Qualify prepared decode counters against clean cohorts")
    parser.add_argument("--decode-identities", action="store_true",
                        help="Retain per-request query geometry in the separate decode counter arm")
    parser.add_argument("--profile", action="store_true",
                        help="Bracket a kernel-only central capture with two clean cohorts")
    parser.add_argument("--profile-state-ms", type=int, default=25)
    parser.add_argument("--profile-decode-buffer-gib", type=int, choices=(1, 2), default=2)
    parser.add_argument("--repeat-main100", type=int, default=1)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--serial-encode-control", action="store_true")
    parser.add_argument("--natural-boundaries", action="store_true")
    parser.add_argument("--natural-hold", action="store_true")
    parser.add_argument("--inline-redirect", action="store_true")
    parser.add_argument("--preissue-crossing-ec", action="store_true",
                        help="Give eligible crossing images a separate one-shot D descriptor")
    parser.add_argument("--preissue-max-suffix-fraction", type=float, default=.35,
                        help="Fractional suffix envelope for crossing expectations")
    parser.add_argument("--preissue-max-suffix-tokens", type=int, default=0,
                        help="Additional absolute suffix envelope for crossing expectations")
    parser.add_argument("--slice-aware-transfer", action="store_true",
                        help="Copy only the visual rows needed by D for the selected cut")
    parser.add_argument("--require-ready-transport", action="store_true",
                        help="Continue on P unless all D suffix image descriptors are already claimable")
    parser.add_argument("--observed-cut-fixtures", action="store_true")
    parser.add_argument("--lead-boundary-control", action="store_true")
    parser.add_argument("--numeric-regime",
                        choices=("bi0", "no-graphs", "bi1", "bi1-p", "bi1-d", "eager", "bi1-eager",
                                 "bi1-dsplit4", "bi1-dsplit8", "bi1-qsplit4"), default="bi0",
                        help="Opt-in P/D numerical diagnostic. E remains BI0.")
    parser.add_argument("--fixed-d-graph-size", type=int, choices=(8,),
                        help="Pad supported D decode CUDA graphs to eight rows while retaining actual Batch/Cap")
    parser.add_argument("--recovery-fixtures", nargs="+", choices=("interior", "image_edge"),
                        help="Run Native A/A and historical teacher-forced divergence diagnostics")
    parser.add_argument("--ratio-gates", nargs="+",
                        choices=("small19", "small64", "small128", "small64_fastjoin", "small19_fastjoin",
                                 "long64_fastjoin", "long128_fastjoin",
                                 "natural05", "natural10", "natural20", "natural30"),
                        help="Run N1 and real staged Mixed N8 for these frozen cut fixtures")
    parser.add_argument("--native-aa", action="store_true",
                        help="Repeat the clean Native cohort in the same start and record exact Greedy agreement")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--stock-cut-trace", action="store_true",
                        help="Observe actual P chunk retirement without changing stock scheduling")
    parser.add_argument("--record-causal-progress", action="store_true",
                        help="Record B-new P features at fixed/old-policy callbacks without affecting admission")
    parser.add_argument("--cost-model", type=Path)
    parser.add_argument("--causal-model", type=Path)
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--gain-estimate", choices=("lower","point"), default="lower")
    parser.add_argument("--qualified-ratios", nargs="+", type=float, choices=(.01,.025,.05,.10,.20,.30))
    parser.add_argument("--qualified-token-cuts", nargs="+", type=int, choices=(19,64,128,256,512,1024,2048,4096))
    parser.add_argument("--ratios", type=float, nargs="+", default=(.30, .20, .10, .05))
    parser.add_argument("--idle-budget-ms", type=float, default=160.)
    parser.add_argument("--safety-ms", type=float, default=10.)
    parser.add_argument("--all-burners-off", action="store_true")
    parser.add_argument("--wall-seconds", type=int, default=2400)
    args = parser.parse_args()
    if bool(args.qps) == args.correctness:
        parser.error("Choose separate correctness or online timing epochs")
    if args.correctness and args.mode != "late10":
        parser.error("The fixed-boundary correctness worker verifies the shared runtime")
    if (args.recovery_fixtures or args.ratio_gates) and not (
        args.correctness and args.serial_encode_control and args.inline_redirect
    ):
        parser.error("Recovery probes require the staged serial E and inline correctness path")
    if args.ratio_gates and args.natural_boundaries:
        parser.error("Ratio tests set forced or natural boundaries per fixture")
    if args.fixed_d_graph_size is not None and (args.numeric_regime in ("eager", "bi1-eager", "no-graphs")
                                                or args.decode_cap != "8"):
        parser.error("Fixed D graph padding requires compiled CUDA-graphs and decode cap 8")
    if args.native_aa and (args.correctness or args.mode not in ("native-matched", "native-common", "native-strong")
                           or args.screen or args.replay):
        parser.error("Native A/A requires clean Native Natural EOS cohorts")
    if args.serial_encode_control and not args.correctness:
        parser.error("Serial Encoder geometry matching is for correctness only")
    if args.natural_hold and not args.natural_boundaries:
        parser.error("A retirement hold requires a natural P boundary")
    if args.inline_redirect and args.mode not in (*LATE_POLICIES, "native-matched"):
        parser.error("Inline EC reassignment is a shared late/matched transport")
    if args.preissue_crossing_ec and not args.inline_redirect:
        parser.error("Crossing EC preissue uses the common inline redirect transport")
    if not 0 <= args.preissue_max_suffix_fraction <= .35 or args.preissue_max_suffix_tokens < 0:
        parser.error("Invalid crossing EC suffix envelope")
    if args.slice_aware_transfer and not (args.inline_redirect and args.mode in LATE_POLICIES):
        parser.error("Slice-aware serving uses the common inline late-bound transport")
    if args.require_ready_transport and not (args.inline_redirect and args.mode in LATE_POLICIES):
        parser.error("The suffix descriptor readiness guard requires a late-bound inline cohort")
    if args.observed_cut_fixtures and not (args.correctness and args.natural_boundaries):
        parser.error("Observed stock cut targets apply to natural-boundary correctness only")
    if args.lead_boundary_control and not (args.serial_encode_control and args.natural_hold):
        parser.error("The staged lead boundary is a serial-E natural-hold correctness control")
    if args.shadow and (args.correctness or args.mode not in LATE_POLICIES):
        parser.error("Shadow mode measures online decision costs with all suffix reservations disabled")
    if args.record_causal_progress and args.mode not in LATE_POLICIES:
        parser.error("P causal progress requires a late-bound callback")
    if args.mode in LATE_POLICIES[1:] and not args.cost_model:
        parser.error("Dynamic ratios require the H200 cost model receipt")
    if (args.causal_model is not None) != (args.mode in CAUSAL_POLICIES):
        parser.error("A causal P-ready calibration model is required for new dynamic policies")
    if args.adaptive_budget and args.mode not in CAUSAL_POLICIES:
        parser.error("An uncapped budget uses the new causal model")
    if args.mode in CAUSAL_POLICIES and not args.shadow and not (args.qualified_ratios or args.qualified_token_cuts):
        parser.error("Active serving requires explicitly qualified suffix geometry")
    if args.qualified_ratios and args.mode not in CAUSAL_POLICIES:
        if args.mode != "late10" or args.qualified_ratios != [.10] or not args.natural_boundaries:
            parser.error("Scoped Late10 uses --natural-boundaries and --qualified-ratios .10")
    if args.idle_budget_ms <= 0 or not 0 <= args.safety_ms < args.idle_budget_ms:
        parser.error("Invalid idle budget or safety margin")
    if args.observe_batch and (args.screen or args.correctness or args.profile):
        parser.error("Choose a clean/counter/clean epoch separately from screening or correctness")
    if args.decode_identities and not (args.screen or args.observe_batch):
        parser.error("Decode identities require a separate counter cohort")
    if args.profile and (args.mode not in ("native-common", "native-strong", "native-matched")
                         or args.screen or args.correctness or len(args.qps) != 1):
        parser.error("Observer qualification requires a separate Native epoch at one load")
    if not 1 <= args.profile_state_ms <= 1000:
        parser.error("Profile state interval must be 1 to 1000 ms")
    if args.repeat_main100 < 1 or (args.repeat_main100 > 1 and args.subset != "all"):
        parser.error("Sustained measurement repeats the complete Main100 roster")
    args.short_roi = args.profile
    args.defer_client_raw = True
    return args


def launch_late_proxy(group, options):
    command = [sys.executable, "-m", f"poc_flashvep.{rt.PACKAGE}.proxy", "--mode", "epd",
               "--name", "epd", "--encoder", "http://127.0.0.1:18840",
               "--prefill", "http://127.0.0.1:18841", "--decoders", "http://127.0.0.1:18842",
               "--port", "18847", "--processor-handoff", "grid_only", "--release-encoder-ipc",
               "--decode-cap", options.decode_cap, "--late-policy", options.mode,
               "--late-idle-budget-ms", str(options.idle_budget_ms), "--late-safety-ms", str(options.safety_ms),
               "--late-ratios", *map(str, options.ratios)]
    if options.cost_model:
        command.extend(["--late-cost-model", str(options.cost_model.resolve(strict=True))])
    if options.causal_model:
        command.extend(["--late-causal-model", str(options.causal_model.resolve(strict=True))])
    if options.adaptive_budget:
        command.append("--late-adaptive-budget")
    if options.gain_estimate == "point":
        command.extend(["--late-gain-estimate", "point"])
    if options.qualified_ratios:
        command.extend(["--late-qualified-ratios", *map(str, options.qualified_ratios)])
    if options.qualified_token_cuts:
        command.extend(["--late-qualified-token-cuts", *map(str, options.qualified_token_cuts)])
    if options.natural_boundaries:
        command.append("--late-natural-boundaries")
    if options.natural_hold:
        command.append("--late-natural-hold")
    if options.shadow:
        command.append("--late-shadow")
    if options.require_ready_transport:
        command.append("--late-require-ready-transport")
    process = group.start("epd", command)
    group.ready(process, 18847, "epd")
    return "http://127.0.0.1:18847"


def run_correctness(args, run, group, url, manifests, ec):
    from .correctness import run_case, run_mixed

    rows = rt.read(manifests[rt.CALIBRATION])["requests"]
    client = rt.module("client")
    asyncio.run(client.execute(rows[:1], [url], "late_correctness_warmup", 1, args.seed, 16, serial=True))
    rt.require(rt.read(run / "state/late_correctness_warmup_complete.json")["passed"], "Late D watcher warmup failed")
    if args.ratio_gates:
        from ..mllm_epd_h200_dynamic_correctness_slice_aware.ratio_gate import run_ratio_gates

        teacher = rt.launch_proxy(group, "teacher", "e_pd", 18848)
        run_ratio_gates(rows, url, teacher, ec, args.numeric_regime, args.ratio_gates,
                        slice_aware=args.slice_aware_transfer)
    if args.recovery_fixtures:
        from ..mllm_epd_h200_dynamic_correctness_slice_aware.recovery import run_recovery

        run_recovery(rows, url, ec, args.numeric_regime, args.recovery_fixtures)
    if args.ratio_gates or args.recovery_fixtures:
        return
    teacher = rt.launch_proxy(group, "teacher", "e_pd", 18848)
    mixed_rows = [rows[index] for index in (2, 0, 5, 6, 12, 4, 11, 9)]
    if args.natural_boundaries and not args.serial_encode_control:
        # The first campaign probe planned B8192 multiples. The follow-up uses stock
        # encoder-aware query endpoints actually witnessed in the first no-hold epoch.
        targets = ({31: 24275, 27: 15947, 20: 16083, 19: 15929, 26: 8192}
                   if args.observed_cut_fixtures or args.inline_redirect
                   else {31: 24576, 27: 16384, 20: 16384, 19: 16384, 26: 8192})
        cases = [("natural_long10", 31, "split"), ("natural_doc10", 27, "split"),
                 ("natural_doc05", 20, "split"), ("natural_multi20", 19, "split"),
                 ("natural_doc30", 26, "split"), ("natural_continue", 31, "continue")]
        report = []
        for label, index, choice in cases:
            cut = targets[index]
            try:
                result = run_case(rows[index], label, cut, url, teacher, choice=choice,
                                  natural="hold" if args.natural_hold else True)
            except Exception as error:
                result = {"case": label, "passed": False, "evidence": "DIAGNOSTIC",
                          "error": f"{type(error).__name__}: {error}"}
            report.append(result)
            print("NATURAL_CORRECTNESS", label, result["passed"], flush=True)
        rt.write(run / "analysis/natural_correctness.json", {"cases": report, "hold": args.natural_hold,
                                                            "fixture_cuts": targets})
        return
    if args.serial_encode_control:
        from .geometry_control import run_geometry_control

        controls = []
        if args.natural_boundaries:
            mixed_rows = [rows[index] for index in (31, 2, 0, 5, 6, 12, 4, 9)]
            rounds = (("lead_interior", .10), ("lead_image_edge", .20)) if args.lead_boundary_control else (
                ("natural10", .10), ("natural20", .20))
            special = {"natural": "hold" if args.natural_hold else True, "use_native_cuts": True}
        else:
            rounds = (("10", .10), ("05", .05), ("20", .20), ("30", .30))
            special = {}
        for code, ratio in rounds:
            tag = f"serial_E_{code}"
            selected = mixed_rows
            options = special
            if args.lead_boundary_control:
                index, cut = (31, 24275) if code == "lead_interior" else (19, 15929)
                selected = [rows[index], *mixed_rows[1:]]
                options = dict(special, lead=True, use_native_cuts=False, targets=[cut] + [None] * 7)
            try:
                result = run_geometry_control(selected, tag, url, teacher, ec, ratio=ratio, **options)
            except Exception as error:
                result = {"passed": False, "evidence": "FAILED_EXECUTION", "error": f"{type(error).__name__}: {error}"}
            controls.append(dict(result, case=tag, ratio=ratio))
            print("SERIAL_E_CONTROL", code, result["passed"], flush=True)
        rt.write(run / "analysis/geometry_control.json", {"cases": controls})
        return
    cases = [("continue_10", 0, .10, "continue"), ("split_10", 0, .10, "split"),
             ("split_05", 4, .05, "split"), ("split_20", 4, .20, "split"),
             ("split_30", 0, .30, "split"), ("image_boundary", 6, None, "split"),
             ("long_document_30", 31, .30, "split")]
    report = []
    for label, index, ratio, choice in cases:
        row = rows[index]
        length = row["total_input_tokens"]
        cut = round(length * (1 - ratio)) if ratio is not None else row["image_visual_spans"][-1][0]
        try:
            result = run_case(row, label, cut, url, teacher, choice=choice)
        except Exception as error:
            result = {"case": label, "passed": False, "evidence": "FAILED_EXECUTION",
                      "error": f"{type(error).__name__}: {error}"}
        report.append(result)
        print("CORRECTNESS", label, result["passed"], flush=True)
    if all(result["passed"] for result in report[:2]):
        for label, ratio in (("mixed_10", .10), ("mixed_05", .05), ("mixed_20", .20), ("mixed_30", .30)):
            single = next(row for row in report if row["case"] == label.replace("mixed", "split"))
            if not single["passed"]:
                continue
            try:
                result = dict(run_mixed(mixed_rows, label, url, teacher, ratio), case=label)
            except Exception as error:
                result = {"case": label, "passed": False, "evidence": "FAILED_EXECUTION",
                          "error": f"{type(error).__name__}: {error}"}
            report.append(result)
            print("CORRECTNESS", label, result["passed"], flush=True)
    rt.write(run / "analysis/correctness.json", {"cases": report})


def main():
    args = arguments()
    worktree = Path(__file__).resolve().parents[2]
    asset_repo, receipt = args.asset_repo.resolve(strict=True), args.receipt.resolve(strict=True)
    run = args.run_dir.resolve()
    rt.git_check(asset_repo)
    runtime = rt.runtime_inventory()
    model, manifests = rt.verify_inputs(asset_repo, receipt)
    inventory = {int(gpu): (uuid, name) for gpu, uuid, name in rt.nvidia("--query-gpu=index,uuid,name")}
    expected = {gpu: inventory[gpu][0] for _, gpu, _, _ in rt.ROLES}
    rt.require(all(inventory[gpu][1] == "NVIDIA H200" for gpu in expected), "H200 layout changed")
    deadline = time.time() + args.wall_seconds
    run.mkdir(mode=0o700)
    for label in ("state", "logs", "raw", "analysis", "data", "configs", "references"):
        (run / label).mkdir()
    if args.profile:
        from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import build_collector
        build_collector(asset_repo, run, Path(sys.executable).parent.parent)
    source, env = configure(worktree, asset_repo, run, deadline, freeze_auxiliary=True)
    late = args.mode in LATE_POLICIES
    matched_transport = late or args.mode == "native-matched"
    auxiliary_dir = source / "poc_flashvep"
    source_hashes = {path.name: rt.sha(path) for path in sorted(
        (auxiliary_dir / Path(__file__).parent.name).glob("*.py")
    )}
    recovery_hashes = {path.name: rt.sha(path) for path in sorted(
        (auxiliary_dir / "mllm_epd_h200_dynamic_correctness_slice_aware").glob("*.py")
    )}
    ec = Path("/dev/shm") / f"h200_dynamic_{env['MLLM_EPD_RUN_NONCE']}"
    workers = os.environ.copy()
    workers.update(MLLM_EPD_METADATA_PATH=str(ec / "processed_prompts"))
    if matched_transport:
        workers["MLLM_EPD_LATE_METADATA"] = "1"
    crossing_envelope = None
    if args.preissue_crossing_ec:
        from ..mllm_epd_prefill_heavy_token_split.crossing_ipc import MAX_FRACTION_ENV, MAX_TOKENS_ENV, PREISSUE_ENV

        max_tokens = max([args.preissue_max_suffix_tokens, *(args.qualified_token_cuts or ())])
        crossing_envelope = {"max_suffix_fraction": args.preissue_max_suffix_fraction, "max_suffix_tokens": max_tokens}
        workers.update({PREISSUE_ENV: "1", MAX_FRACTION_ENV: str(args.preissue_max_suffix_fraction),
                        MAX_TOKENS_ENV: str(max_tokens), "MLLM_EPD_EC_PATH": str(ec)})
    role_environment = {}
    if late:
        role_environment["prefill"] = {"MLLM_EPD_LATE_DECISION_URL": "http://127.0.0.1:18847/campaign/late-decision"}
    if args.stock_cut_trace:
        for role in ("prefill", "decode"):
            role_environment.setdefault(role, {})["MLLM_EPD_STOCK_CUT_TRACE"] = "1"
    if args.mode in CAUSAL_POLICIES or args.record_causal_progress:
        role_environment["prefill"]["MLLM_EPD_CAUSAL_SLACK"] = "1"
    if args.inline_redirect:
        role_environment.setdefault("decode", {})["MLLM_EPD_LATE_INLINE_REDIRECT"] = "1"
    if args.slice_aware_transfer:
        role_environment["decode"]["MLLM_EPD_SLICE_AWARE_EC"] = "1"
    invariance_roles = {
        "bi1": ("prefill", "decode"), "bi1-eager": ("prefill", "decode"),
        "bi1-p": ("prefill",), "bi1-d": ("decode",),
        "bi1-dsplit4": ("prefill", "decode"), "bi1-dsplit8": ("prefill", "decode"),
        "bi1-qsplit4": ("prefill", "decode"),
    }
    for role in invariance_roles.get(args.numeric_regime, ()):
        role_environment.setdefault(role, {})["VLLM_BATCH_INVARIANT"] = "1"
    if args.numeric_regime.startswith("bi1-dsplit"):
        role_environment["decode"]["MLLM_EPD_BI1_D_SPLITS"] = args.numeric_regime.removeprefix("bi1-dsplit")
    if args.numeric_regime == "bi1-qsplit4":
        for role in ("prefill", "decode"):
            role_environment[role]["MLLM_EPD_BI1_D_SPLITS"] = "4"
    if args.decode_identities:
        role_environment.setdefault("decode", {})["MLLM_EPD_DECODE_COUNTER_IDENTITIES"] = "1"
    if args.correctness:
        workers["MLLM_EPD_LATE_CORRECTNESS"] = "1"
    if args.serial_encode_control:
        workers["MLLM_EPD_LATE_VERIFY_E_SERIAL"] = "1"
    rt.write(run / "state/model_download_complete.json", {"directory": str(model), "revision": rt.MODEL_REVISION})
    shutil.copyfile(manifests[rt.CALIBRATION], run / "data" / rt.CALIBRATION)
    rt.write(run / "state/environment.json", {
        "runtime": runtime, "receipt_SHA256": rt.sha(receipt), "source_SHA256": fingerprint_source(source),
        "runner_SHA256": source_hashes, "recovery_source_SHA256": recovery_hashes,
        "authorized_uuid_by_gpu": expected, "argv": sys.argv,
        "git_head": subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip(),
        "EC_inline_P": not matched_transport or args.inline_redirect, "EC_inline_redirect": args.inline_redirect,
        "late": late, "shadow": args.shadow, "numeric_regime": args.numeric_regime,
        "fixed_d_graph_size": args.fixed_d_graph_size,
        "cost_model_SHA256": rt.sha(args.cost_model) if args.cost_model else None,
        "causal_model_SHA256": rt.sha(args.causal_model) if args.causal_model else None,
        "slice_aware_transfer": args.slice_aware_transfer, "deadline": deadline,
        "crossing_preissue_envelope": crossing_envelope,
    })
    group = rt.OwnedProcesses(run, workers, source, deadline)

    def cancelled(*_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("Dynamic experiment cancelled or expired")

    signal.signal(signal.SIGALRM, cancelled)
    signal.signal(signal.SIGTERM, cancelled)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
    try:
        rt.write(run / "state/burners_before.json", burner_status())
        rt.write(run / "state/host.json", snapshot(topology=True))
        scope = {gpu: value[0] for gpu, value in inventory.items()} if args.all_burners_off else expected
        with experiment_lease(scope, run / "state") as stopped:
            gpus, occupancy = rt.verify_devices(expected)
            rt.write(run / "state/lease.json", {"stopped_burners": stopped, "gpu_inventory": gpus,
                                                "foreign_gpu_occupancy": occupancy})
            rt.verify_ports()
            ec.mkdir(mode=0o700)
            try:
                rt.launch_models(group, args.mode, ec, expected, asset_repo / rt.REPORT / "media",
                                 token=matched_transport or args.mode == "idle10", common=args.mode != "native-strong",
                                 role_environment=role_environment, eager_pd=args.numeric_regime.endswith("eager"),
                                 no_graphs_pd=args.numeric_regime == "no-graphs",
                                 fixed_d_graph_size=args.fixed_d_graph_size)
                if matched_transport and not args.inline_redirect:
                    result = rpc(18840, "ec_ipc/inline_policy", {"enabled": False})
                    rt.write(run / "state/on_demand_EC.json", result)
                if late:
                    rt.write(run / "state/D_watcher_initial.json", rpc(18842, "runtime_state", {}))
                    url = launch_late_proxy(group, args)
                else:
                    url = rt.launch_proxy(group, "epd", "epd", 18847, token=args.mode == "idle10",
                                          decode_cap=args.decode_cap, idle=args.mode == "idle10")
                if args.correctness:
                    run_correctness(args, run, group, url, manifests, ec)
                else:
                    measured_cohorts(args, run, url, manifests)
                    if args.stock_cut_trace:
                        rt.write(run / "state/stock_cut_trace_flush.json", {
                            role: rpc(port, "stock_cut_trace/flush", {})
                            for role, port in (("prefill", 18841), ("decode", 18842))})
                if args.numeric_regime.startswith(("bi1-dsplit", "bi1-qsplit")):
                    fixed = {}
                    for role, port in (("prefill", 18841), ("decode", 18842)):
                        if "MLLM_EPD_BI1_D_SPLITS" in role_environment[role]:
                            worker, = rt.http_json(port, "campaign/worker_info", timeout=30)
                            fixed[role] = worker["numerics"]["fixed_single_query_FA3"]
                    rt.require(all(item["metadata_uses"]["single_query_fixed"] > 0 for item in fixed.values()),
                               "Fixed single-query split count was never used on a configured worker")
                    rt.write(run / "state/fixed_decode_attention.json", fixed)
                if args.preissue_crossing_ec:
                    registry, = rt.http_json(18840, "campaign/ec_ipc/state", timeout=30)
                    rt.write(run / "state/crossing_registry_closeout.json", registry)
                    rt.require(registry["live_images"] == registry["outstanding_exports"] == 0
                               and registry["crossing_issued"] == registry["crossing_acknowledged"]
                               + registry["crossing_revoked"], "Crossing descriptor closeout is incomplete")
            finally:
                group.close()
                rt.verify_devices(expected)
                shutil.rmtree(ec)
    except BaseException as error:
        rt.write(run / "state/failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rt.write(run / "state/complete.json", {"status": "COMPLETE", "time": time.time()})


if __name__ == "__main__":
    main()
