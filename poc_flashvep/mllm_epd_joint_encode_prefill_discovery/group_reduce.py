"""Join physical E group receipts, actual LM retirement and fixed-arm latency."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import tarfile

from ..mllm_epd_modality_aware_idle_execution_discovery.readiness_execution import ARMS


def records(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        if line.endswith("\n"):
                            raise
                        return


def ms(end, start):
    return (end - start) / 1e6


def summarize(run, cases):
    by_image, signed, retired = {}, defaultdict(dict), defaultdict(list)
    for group in records(run / "raw/group_encoder.jsonl"):
        for image in group["ids"]:
            if image in by_image:
                raise ValueError("Image has multiple native producer groups")
            by_image[image] = group
    for path in sorted((run / "raw").glob("ready_retired_*.jsonl")):
        for row in records(path):
            retired[row["key"]].append(row)
    fixtures = {f["case_id"]: f for f in json.loads(cases.read_bytes())["cases"]}
    import numpy as np

    prompts = {}
    for case, fixture in fixtures.items():
        with np.load(fixture["request"]["token_metadata_file"]) as source:
            text = json.dumps(source["input_ids"].ravel().tolist(), separators=(",", ":"))
        prompts[case] = hashlib.sha256(text.encode()).hexdigest()
    rows, paired = [], defaultdict(dict)
    for trial in records(run / "raw/group_trials.jsonl"):
        case, arm, mode = (trial[k] for k in ("fixture", "arm", "group_mode"))
        fixture = fixtures[case]
        if trial["prompt_token_ids_SHA256"] != prompts[case]:
            raise ValueError("Serving prompt token IDs differ from original input")
        spans = fixture["request"]["image_visual_spans"]
        proof = sorted(retired[trial["key"]], key=lambda row: row["retire_ns"])
        if not proof or proof[-1]["retired_rows"] != trial["prompt_tokens"]:
            raise ValueError(f"Missing original prompt retirement: {trial['tag']}")
        for a,b in zip(proof, proof[1:]):
            if a["retired_rows"] != b["before"]:
                raise ValueError(f"Non-contiguous LM execution: {trial['tag']}")
        images = trial["image_ids"]
        groups = []
        if mode != "stock":
            for key in images:
                group = by_image[key]
                if group["mode"] != mode:
                    raise ValueError("Unexpected CUDA publication group mode")
                if not groups or groups[-1] != group:
                    groups.append(group)
            if [key for g in groups for key in g["ids"]] != images:
                raise ValueError("Group publisher changed original image order")
        all_lower = max((g["cuda_ready_lower_ns"] for g in groups), default=None)
        all_upper = max((g["cuda_ready_upper_ns"] for g in groups), default=None)
        required = groups and max(by_image[i]["cuda_ready_upper_ns"] for i in images[:trial["prefix_images"]])

        def visual(deadline):
            indices = [r["retired_rows"] for r in proof if deadline is not None and r["retire_ns"] < deadline]
            end = max(indices, default=0)
            return sum(max(0, min(b,end) - a) for a,b in spans if a < end)

        first_visual = next(r["retire_ns"] for r in proof if r["retired_rows"] > spans[0][0])
        raw = {"group_mode": mode, "fixture": case, "arm": arm, "repeat": trial["repeat"],
               "E_HTTP_ms": ms(trial["e_ack_ns"],trial["start_ns"]),
               "descriptor_or_CUDA_marker_lead_ms": ms(trial["all_published_ns"],trial["prefix_published_ns"]),
               "E_group_CUDA_elapsed_ms": sum(g["gpu_elapsed_ms"] for g in groups) if groups else None,
               "last_descriptor_to_CUDA_ms": ms(groups[-1]["cuda_ready_upper_ns"],groups[-1]["native_saved_ns"])
                   if groups else None,
               "E_first_group_CUDA_ready_ms": ms(groups[0]["cuda_ready_upper_ns"],trial["start_ns"]) if groups else None,
               "E_all_CUDA_ready_upper_ms": ms(all_upper,trial["start_ns"]) if all_upper is not None else None,
               "E_legal_prefix_CUDA_ready_ms": ms(required,trial["start_ns"]) if groups else None,
               "conservative_E_ready_lead_ms": ms(all_lower,required) if groups else None,
               "first_visual_after_E_ACK_ms": ms(first_visual,trial["e_ack_ns"]),
               "visual_rows_before_all_CUDA_lower": visual(all_lower) if groups else None,
               "visual_rows_before_E_ACK": visual(trial["e_ack_ns"]),
               "visual_rows_before_final_group_submit": visual(groups[-1]["submit_ns"]) if groups else None,
               "TTFT_ms": trial["TTFT_ms"], "TPOT_ms": trial["TPOT_ms"],
               "ITL_max_ms": max(trial["ITL_ms"]), "absolute_decode_pass": trial["TPOT_ms"] <= 33 and max(trial["ITL_ms"]) <= 34}
        rows.append(raw)
        arms = paired[(mode,case,trial["repeat"])]
        if arm in arms:
            raise ValueError(f"Duplicate measured quartet arm: {trial['tag']}")
        arms[arm] = (raw, trial["output_token_ids"])
        raw["timeline"] = [{"submit_ms": ms(g["submit_ns"],trial["start_ns"]),
                            "native_saved_ms": ms(g["native_saved_ns"],trial["start_ns"]),
                            "ready_ms": ms(g["cuda_ready_upper_ns"],trial["start_ns"]),
                            "images": len(g["ids"]), "visual_rows": g["visual_rows"]} for g in groups]
        raw["first_visual_retired_ms"] = ms(first_visual,trial["start_ns"])
    for trial in records(run / "raw/group_signatures.jsonl"):
        signed[trial["fixture"]][trial["group_mode"]] = trial["encoder_signatures"]
    correctness = {}
    for case, modes in signed.items():
        base = modes["stock"]
        correctness[case] = {mode: {"images": len(base), "bitwise": sum(a == b for a,b in zip(base,value,strict=True))}
                             for mode,value in modes.items()}
    quartets = []
    for (mode,case,repeat), trial in sorted(paired.items()):
        if set(trial) != set(ARMS):
            continue
        lp,ep,ld,ed = (trial[arm][0]["TTFT_ms"] for arm in ARMS)
        stock = paired.get(("stock",case,repeat), {})
        original_ttft = stock["late_p"][0]["TTFT_ms"] if "late_p" in stock else None
        quartets.append({"mode": mode, "case": case, "repeat": repeat, "early_p_save_ms": lp-ep,
                         "early_d_save_ms": ld-ed, "d_increment_ms": (ld-ed)-(lp-ep),
                         "late_p_TTFT_ms": lp, "early_p_TTFT_ms": ep, "late_d_TTFT_ms": ld, "early_d_TTFT_ms": ed,
                         "early_p_vs_stock_late_p_save_ms": original_ttft-ep if original_ttft is not None else None,
                         "early_d_vs_stock_late_p_save_ms": original_ttft-ed if original_ttft is not None else None,
                         "same_p_outputs": trial["late_p"][1] == trial["early_p"][1],
                         "same_d_outputs": trial["late_d"][1] == trial["early_d"][1],
                         "same_four_outputs": len({tuple(v[1]) for v in trial.values()}) == 1})
    modes = list(dict.fromkeys(r["group_mode"] for r in rows))
    summary = {}
    for mode in modes:
        indices = [r for r in rows if r["group_mode"] == mode]
        early = [r for r in indices if r["arm"].startswith("early")]
        quads = [q for q in quartets if q["mode"] == mode]
        data = {"trials": len(indices), "quartets": len(quads), "decode_passes": sum(r["absolute_decode_pass"] for r in indices),
                "early_retired_gpu": sum((r["visual_rows_before_all_CUDA_lower"] or 0) > 0 for r in early),
                "early_retired_before_ack": sum(r["visual_rows_before_E_ACK"] > 0 for r in early),
                "early_retired_before_last_group": sum((r["visual_rows_before_final_group_submit"] or 0) > 0 for r in early)}
        data.update({f"{metric}_p50": median(n) if (n := [r[metric] for r in indices if r[metric] is not None]) else None
                     for metric in ("E_HTTP_ms","E_group_CUDA_elapsed_ms","E_first_group_CUDA_ready_ms",
                                    "E_all_CUDA_ready_upper_ms","conservative_E_ready_lead_ms","last_descriptor_to_CUDA_ms")})
        data.update({f"{col}_p50": median(vals) if (vals := [q[col] for q in quads if q[col] is not None]) else None
                     for col in ("early_p_save_ms","early_d_save_ms","d_increment_ms",
                                 "early_p_vs_stock_late_p_save_ms","early_d_vs_stock_late_p_save_ms")})
        summary[mode] = data
    return {"scope": "Serial native H200 physical trials. Group ready bounds are CUDA stream end-event witnesses. "
                     "Early retired rows are actual GPU results; they do not enumerate simultaneous kernels.",
            "trials": rows, "quartets": quartets, "by_mode": summary, "encoder_exact": correctness}


def figures(value, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(exist_ok=True)
    modes = value["by_mode"]
    whole = "awhole" if "awhole" in modes else "whole"
    eligible = [k for k,v in modes.items() if k not in {"stock",whole} and v["early_retired_gpu"]]
    chosen = min(eligible,key=lambda k:modes[k]["E_HTTP_ms_p50"]) if eligible else next(iter(modes), None)
    examples = [next((row for row in value["trials"] if row["group_mode"]==k and row["arm"]=="early_p"),None)
                for k in (whole,chosen)]
    if all(examples):
        fig,ax = plt.subplots(figsize=(9,3.7),layout="constrained")
        for y,row in enumerate(examples):
            for group in row["timeline"]:
                ax.broken_barh([(group["submit_ms"],group["ready_ms"]-group["submit_ms"])], (y-.18,.36),
                               facecolors="#397798",alpha=.8)
            ax.plot(row["first_visual_retired_ms"],y,marker="D",markersize=6,color="#bb451f")
        ax.set(yticks=[0,1],yticklabels=[f'E group {row["group_mode"]}' for row in examples],
               xlabel="Milliseconds from request submission", title="E group launch to CUDA-ready and P’s first visual retirement")
        fig.savefig(directory/"group_cuda_ready_timeline.svg")
        plt.close(fig)
    ready_samples = [r for r in value["trials"] if r["last_descriptor_to_CUDA_ms"] is not None]
    if ready_samples:
        mode_set = list(dict.fromkeys(r["group_mode"] for r in ready_samples))
        gaps = [[r["last_descriptor_to_CUDA_ms"] for r in ready_samples if r["group_mode"] == mode]
                for mode in mode_set]
        fig, ax = plt.subplots(figsize=(8,3.7),layout="constrained")
        ax.boxplot(gaps, tick_labels=mode_set)
        ax.set(xlabel="E image group", ylabel="Native export return to CUDA-ready confirmation (ms)",
               title="Last image group, paired physical measurements")
        fig.savefig(directory/"group_descriptor_cuda_gap.svg")
        plt.close(fig)
    mode_set = [k for k,v in modes.items() if v["early_p_save_ms_p50"] is not None]
    if mode_set:
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
        for arm,label in (("early_p","Early P"),("early_d","Early D")):
            axes[0].plot(mode_set,[modes[k][f"{arm}_save_ms_p50"] for k in mode_set],"o-",label=label)
        axes[0].axhline(0,color="grey",lw=1)
        axes[0].set(xlabel="E image group",ylabel="Paired median TTFT saving (ms)")
        axes[0].legend(frameon=False)
        axes[1].plot(mode_set,[modes[k]["E_HTTP_ms_p50"] for k in mode_set],"o-",label="E HTTP")
        axes[1].plot(mode_set,[modes[k]["E_group_CUDA_elapsed_ms_p50"] for k in mode_set],"s-",label="E CUDA stream intervals")
        axes[1].plot(mode_set,[modes[k]["E_all_CUDA_ready_upper_ms_p50"] for k in mode_set],"^-",label="All E groups CUDA ready")
        axes[1].set(xlabel="E image group",ylabel="Median time (ms)")
        axes[1].legend(frameon=False)
        fig.savefig(directory/"group_early_recipient_comparison.svg")
        plt.close(fig)


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_raw(run, cases, out):
    raw = run / "raw"
    paths = {path for mask in ("group_*.jsonl", "ready_retired_*.jsonl", "group_*_profile_receipts.json")
             for path in raw.glob(mask)}
    paths.update(path for path in (raw / "profiles" / "group_kernel_overlap").rglob("*") if path.is_file())
    run_metadata = ("configs/frozen_plan.json", "state/environment.json", "state/warmup_complete.json",
                    "state/groups_warmup_complete.json", "state/groups_final_ownership.json")
    paths.update(run / name for name in run_metadata if (run / name).is_file())
    archive = out / "raw_groupwise.tar.gz"
    manifest = {}
    with archive.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as packed:
        with tarfile.open(fileobj=packed, mode="w") as bundle:
            for path in sorted(paths):
                name = str(path.relative_to(run))
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open("rb") as stream:
                    bundle.addfile(info, stream)
                manifest[name] = {"sha256": hash_file(path), "bytes": info.size}
    evidence = {"run_directory": str(run), "original_fixture_file": {"path": str(cases), "sha256": hash_file(cases)},
                "raw_groupwise_tar_gz_sha256": hash_file(archive), "files": manifest}
    (out / "raw_evidence_manifest.json").write_text(json.dumps(evidence,indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--figure",action="store_true")
    parser.add_argument("--archive",action="store_true",help="Package a closed run's raw receipts, trace and hashes")
    args = parser.parse_args()
    data = summarize(args.run,args.cases)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"group_summary.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    if data["trials"]:
        keys = [k for k in data["trials"][0] if k != "timeline"]
        with (args.out/"group_trials.csv").open("w",newline="") as target:
            w=csv.DictWriter(target,fieldnames=keys,extrasaction="ignore",lineterminator="\n")
            w.writeheader()
            w.writerows(data["trials"])
    if args.figure:
        figures(data,args.out/"figures")
    if args.archive:
        archive_raw(args.run,args.cases,args.out)
    print(json.dumps(data["by_mode"],indent=2))


if __name__ == "__main__":
    main()
