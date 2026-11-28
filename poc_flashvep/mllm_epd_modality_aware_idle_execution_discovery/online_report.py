"""Render fresh clean serving comparisons and historical controls."""

from .online_evidence import NATIVE_ROUTES, flatten, name


def number(value):
    return "—" if value is None else f"{value:.3f}"


def percentage(value):
    return "—" if value is None else f"{value:+.2%}"


def online_table(records):
    header = ("| Case | QPS / output | P/D token budget | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | "
              "Decode passes | Joint passes | Decode / joint goodput req/s | D-local requests |")
    table = [header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        flat = flatten(row)
        n = row["planned"]
        ttft = flat["TTFT_ms_p95"]
        values = [
            name(row), f"{flat['qps']:g} / " + ("EOS" if flat["output"] is None else str(flat["output"])),
            f"{flat['p_budget']} / {flat['d_budget']}",
            f"{row['completed']}/{n}", number(ttft / 1000 if ttft is not None else None),
            number(flat["TPOT_ms_p95"]), number(flat["ITL_ms_p99"]), f"{flat['absolute_decode_pass']}/{n}",
            f"{flat['joint_pass']}/{n}",
            f"{number(flat['decode_qualified_goodput_req_s'])} / {number(flat['joint_goodput_req_s'])}",
            str(flat["D_local_completed_requests"]) if flat["proxy_routes_joined"] else "—",
        ]
        table.append("| " + " | ".join(values) + " |")
    return "\n".join(table) if records else "No completed client-facing traces were available."


def comparison_table(pairs):
    table = [
        "| Native → extra work | Start | Δ TTFT p95 | Δ TPOT p95 | Δ ITL p99 | Δ E2E p95 | Δ joint goodput | Δ Decode goodput | Exact prompt / output | Normalized text exact | Strict Decode |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in pairs:
        deltas = row["candidate_over_baseline_relative_change"]
        observed_text = row["paired_response_text_receipts"]
        values = [f"{row['baseline']} → {row['candidate']}", "same" if row["same_server_start"] else "separate",
                  *[percentage(deltas[k]) for k in ("TTFT_ms", "TPOT_ms", "ITL_ms", "E2E_ms", "joint_goodput_req_s",
                                                   "decode_qualified_goodput_req_s")],
                  f"{row['full_prompt_exact_requests']}/{row['planned']} / "
                  f"{row['full_output_exact_requests']}/{row['planned']}",
                  f"{row['normalized_response_text_exact_count']}/{observed_text}" if observed_text else "—",
                  row["strict_decode_verdict"]]
        table.append("| " + " | ".join(values) + " |")
    return "\n".join(table) if pairs else "No additional-work arm had a measured matching Native arm."


def markdown(fresh, historical, pairs, topology, text_replays, stages, census, figures):
    clean = [row for row in fresh if row["evidence"] == "MEASURED_ONLINE" and row["measurement_role"] == "SERVING"]
    previous = [row for row in historical if row["evidence"] == "MEASURED_ONLINE"]
    paragraphs = [
        "# Decode-protected online serving",
        "Finite-cohort open-loop metrics use the first planned arrival through the last recorded drain. "
        "Decode attainment requires TPOT ≤33 ms and every ITL ≤34 ms. Joint attainment also requires TTFT ≤5 s. "
        "All planned requests remain in attainment denominators. Latency percentiles describe completed requests. "
        "ITL percentiles pool recorded token gaps. A one-token response has TPOT zero under the client's contract "
        "and contributes no gap sample.",
        "## Fresh clean client-facing arms",
        online_table(clean),
        "## Matched Native and additional-work arms",
        "A strict pair requires complete cohorts, the same model, physical GPU assignment, budgets and transfer policy, "
        "and identical prompt digests and seeded arrivals. The Decode gate requires TPOT p95 and ITL p99 within +5%, "
        "at least 95% absolute Decode attainment, and at least the Native Decode pass count. "
        "Normalized text equality uses the originally streamed text after Unicode NFKC, case folding and whitespace "
        "collapse. Whole generated tokens and normalized streamed text are identity checks. Candidate task-score "
        "noninferiority evidence status is NOT_RUN.",
        comparison_table(pairs),
        "## Historical DP3 controls",
        "The prior H200 DP3 runs use the same number of GPUs, each with a complete model and separate KV budget. "
        "They retain their own runtime starts and warmup. Their joint goodput and tails contextualize the "
        "serving trade-off. Strict paired gates use fresh Native arms.",
        online_table(previous),
    ]
    if topology:
        paragraphs.extend(["## Fresh matched DP3 controls",
                           "These arms retain the original request cohort and arrival schedule on the same physical GPUs. "
                           "DP3 uses different operator and communication paths and its own B2048 batch budget. "
                           "The paired changes describe a topology comparison. Its strict same-runtime gate is ineligible.",
                           comparison_table(topology)])
    if census:
        historical_census = [row for row in census if row["scope"] == "HISTORICAL_NATIVE"]
        fresh_census = [row for row in census if row["scope"] == "FRESH"]
        sizes = ", ".join(f"N{n}" for n in sorted({row["planned"] for row in fresh_census}))
        paragraphs.extend(["## Native Fixed16 persistence",
                           f"Fresh {sizes} reuse the 24 original Very-long requests at 1.465 offered req/s, "
                           "so the larger finite cohorts test the longer pressure interval over the same inputs. "
                           "The N24 point is historical.",
                           online_table(historical_census)])
    heldout = [row for row in clean if row["setup"].get("cohort_id") == "heldout" and row["setup"]["output"] is None]
    if heldout:
        table = ["| Held-out Natural EOS run | Emitted tokens, min / p50 / max | Finish reasons |",
                 "|---|---:|---|"]
        for row in heldout:
            lens = row["output_length"]
            reason = ", ".join(f"{key} {n}" for key, n in sorted(row["finish_reasons"].items()))
            table.append(f"| {name(row)} | {lens['min']} / {lens['p50']} / {lens['max']} | {reason} |")
        paragraphs.extend(["## Held-out Natural EOS output agreement",
                           "The independent original requests use ordinary model generation with the serving cap. "
                           "Requested output mode and actual stopping evidence are reported together. "
                           "Full prompt/token identities and first divergent steps are in the paired receipt. "
                           "This evidence has no scored task comparison.", "\n".join(table)])
    text_controls = [row for row in fresh if row["measurement_role"] == "TEXT_CONTROL"
                     and row["evidence"] == "MEASURED_ONLINE"]
    if text_controls:
        paragraphs.extend([
            "## Text-only arrival controls",
            "The text-only arms use the same language backbone and a similar LM length to the Very-long image workload. "
            "Their visible inputs are distinct from the image prompts. The baseline and replay probe the arrival stream "
            "and decode tail descriptively.",
            online_table(text_controls),
        ])
        for rec in text_replays:
            source = rec["native_source"]
            if rec["replayed_source_ready_events_match"] and rec["actual_replay_planned_schedule_matches"]:
                paragraphs.append(f"{rec['text_replay']} uses {rec['native_source_event_count']} actual full-E-ready "
                                  f"events from {source}. The event identities, frozen offsets and client scheduled "
                                  "arrival offsets reproduce the recorded source.")
            else:
                paragraphs.append(f"{rec['text_replay']} has unresolved source readiness or scheduled-arrival matching.")
            if rec.get("text_poisson"):
                cadence = number(rec["observed_finite_native_ready_rate_req_s"])
                poisson = rec["text_poisson_nominal_offered_QPS"]
                paragraphs.append(f"The observed finite encoder-ready cadence is {cadence} req/s, using the intervals "
                                  f"from first to last event. The independent Poisson nominal offered rate is {poisson:g} req/s.")
                paragraphs.append(f"Matched text output IDs agree on {rec['whole_token_sequences_equal']}/"
                                  f"{rec['aligned_text_inputs']} prompts across Poisson and readiness replay. "
                                  f"Offered QPS differed by {percentage(rec['relative_replay_qps_vs_text_poisson'])}.")
    if stages:
        table = ["| Stage reference | Stage | Requests | Requests/s | Token rate and field | Batch-one |",
                 "|---|---|---:|---:|---|---|"]
        for row in stages:
            fields = {cohort["token_field"] for cohort in (row["cohorts"] or []) if cohort.get("token_field")}
            domain = ", ".join(sorted(fields)) if fields else "token field unavailable"
            values = [name(row), str(row["stage"]), str(row["requests"]), number(row["requests_per_second"]),
                      f"{number(row['tokens_per_second'])}/s ({domain})", str(row["batch_one"])]
            table.append("| " + " | ".join(values) + " |")
        paragraphs.extend(["## Independent stage references",
                           "These measurements come from independent stage service receipts. They describe the measured "
                           "stage contract and do not contribute to the online request goodput.",
                           "\n".join(table)])
    observers = [row for row in fresh if row["instrumentation_family"]]
    if observers:
        paragraphs.extend(["## Observer clean controls and profiles",
                           "Observer clean controls are retained in their own series. Instrumented arms retain "
                           "diagnostic scope, including any failed requests during that arm."])
        for family, title in (("CUDA_GRAPH_CUPTI", "CUDA Graph observer series"),
                              ("CUPTI_KERNEL", "Kernel-only observer series")):
            members = [row for row in observers if row["instrumentation_family"] == family]
            if not members:
                continue
            paragraphs.extend([f"### {title}", online_table(
                [row for row in members if row["evidence"] in {"MEASURED_ONLINE", "DIAGNOSTIC"}])])
            withheld = [f"{name(row)} ({row['evidence']})" for row in members
                        if row["evidence"] not in {"MEASURED_ONLINE", "DIAGNOSTIC"}]
            if withheld:
                paragraphs.append("Arms without a completion receipt: " + ", ".join(withheld) + ".")
    pending = [f"{name(row)} ({row['evidence']})" for row in fresh
               if row["measurement_role"] == "SERVING" and row["evidence"] != "MEASURED_ONLINE"]
    paired = {row[key] for row in pairs for key in ("baseline", "candidate")}
    unpaired = [name(row) for row in clean if name(row) not in paired
                and (row["setup"]["count"] >= 128 or row["setup"]["output"] is None)]
    paragraphs.append("## Evidence scope")
    if pending:
        paragraphs.append("Cohorts without finished receipts: " + ", ".join(pending) + ".")
    if unpaired:
        paragraphs.append("Larger-census or Natural EOS observations without a matched intervention: "
                          + ", ".join(unpaired) + ".")
    if not heldout:
        paragraphs.append("Held-out Natural EOS evidence status is NOT_RUN in these receipts.")
    paragraphs.append("Finite-cohort throughput measures completion and drain at the offered load. "
                      "Independent paired restarts, sustainable QPS and candidate task-score noninferiority require separate evidence.")
    prefix = "../mllm_epd_modality_aware_idle_execution_discovery/results"
    links = [f"[All tails, pass counts and rates]({prefix}/online_cases.csv)",
             f"[Paired measurements and raw SHA-256 receipts]({prefix}/online_evidence.json)",
             "[Reduction source](../mllm_epd_modality_aware_idle_execution_discovery/online_evidence.py)"]
    links.extend(f"[{title}]({prefix}/figures/{filename})" for title, filename in figures)
    paragraphs.append(" · ".join(links))
    return "\n\n".join(paragraphs) + "\n"


def plot_persistence(plt, census, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    old = [row for row in census if row["scope"] == "HISTORICAL_NATIVE"]
    fresh = sorted((row for row in census if row["scope"] == "FRESH"),
                   key=lambda row: (row["planned"], row["tag"]))
    first_by_census = {}
    for row in fresh:
        first_by_census.setdefault(row["planned"], row)
    primary = list(first_by_census.values())
    repeats = [row for row in fresh if row not in primary]

    def x(rows):
        return [row["planned"] for row in rows]

    def ttft(rows):
        return [row["metrics_ms"]["TTFT_ms"]["p95"] / 1000 for row in rows]

    def achievement(rows, criterion):
        return [row["attainment_counts"][criterion] * 100 / row["planned"] for row in rows]

    for row in old:
        axes[0].scatter(x([row]), ttft([row]), marker="D", facecolors="none", edgecolors="#526171",
                        label="Historical Native N24")
        for criterion, color in (("joint", "#1e5f88"), ("absolute_decode", "#427448")):
            axes[1].scatter(x([row]), achievement([row], criterion), marker="D", facecolors="none", edgecolors=color)

    axes[0].plot(x(primary), ttft(primary), "o-", color="#1e5f88", label="Fresh Native")
    if repeats:
        axes[0].scatter(x(repeats), ttft(repeats), marker="o", facecolors="none", edgecolors="#1e5f88",
                        label="Fresh independent start")
    for criterion, label, color in (("joint", "Joint", "#1e5f88"), ("absolute_decode", "Decode", "#427448")):
        axes[1].plot(x(primary), achievement(primary, criterion), "o-", color=color, label=label)
        axes[1].scatter(x(repeats), achievement(repeats, criterion), marker="o", facecolors="none", edgecolors=color)
    axes[0].set_ylabel("TTFT p95 (s)")
    axes[1].set_ylabel("Request attainment (%)")
    axes[1].set_ylim(-3, 104)
    for ax in axes:
        ax.set_xlabel("Finite request census")
        ax.set_xticks(sorted(set(x(census))))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Native Very-long Fixed16 · 1.465 offered req/s · historical N24 shown as an open diamond")
    fig.savefig(out / "online_persistence.svg", metadata={"Date": None})
    plt.close(fig)


def plot(fresh, historical, pairs, topology, census, out):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "epd-idle-online-v1"
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    figures = []
    if census:
        plot_persistence(plt, census, out)
        figures.append(("Native census persistence", "online_persistence.svg"))
    rows = {name(row): row for row in fresh}
    comparable = [pair for pair in pairs if pair["strict_gate_eligible"]]
    topology = [pair for pair in topology if pair["same_GPU_and_model"] and pair["both_complete"]]
    if not comparable and not topology:
        return figures
    keys = ("cohort_id", "count", "qps", "output")

    def group_key(row):
        return tuple(row["setup"][key] for key in keys)

    groups = {}
    for pair in comparable + topology:
        base, test = (rows[pair[key]] for key in ("baseline", "candidate"))
        group = groups.setdefault(group_key(base), {})
        group.update({name(row): row for row in (base, test)})
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 4.2 * len(groups)), squeeze=False, layout="constrained")
    for ax, ((cohort, count, qps, output), group) in zip(axes.flat, groups.items(), strict=True):
        prior = [row for row in historical if row.get("complete")
                 and group_key(row) == (cohort, count, qps, output)]
        points = []
        for row in list(group.values()) + prior:
            itl = row["metrics_ms"]["ITL_ms"]["p99"]
            rate = row["joint_goodput_req_s"]
            if itl is None or rate is None:
                continue
            old = row["scope"] == "HISTORICAL_DP3"
            dp = row["setup"]["route"] == "dp3"
            color = "#526171" if old else (
                "#4d7655" if dp else "#1e5f88" if row["setup"]["route"] in NATIVE_ROUTES else "#bf5426")
            label = f"Prior DP3 B{row['setup']['p_budget']}" if old else row["tag"]
            attained = row["attainment_counts"]["absolute_decode"]
            ax.scatter(itl, rate, marker="D" if dp else "o", s=55, facecolor="none" if old else color,
                       edgecolor=color, label=f"{label} ({attained}/{count} Decode)")
            if not old:
                points.append((itl, rate))
        frontier = sorted({(x, y) for x, y in points
                           if not any(a <= x and b >= y and (a < x or b > y) for a, b in points)})
        if len(frontier) > 1:
            ax.plot([x for x, _ in frontier], [y for _, y in frontier], color="#777777",
                    linestyle=":", linewidth=1, label="Fresh two-metric frontier")
        output_text = "Natural EOS" if output is None else f"Fixed{output}"
        ax.set(xscale="log", xlabel="ITL p99 (ms, log scale)", ylabel="Joint SLO goodput (req/s)",
               title=f"{cohort} · N{count} · offered QPS {qps:g} · {output_text}")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=.2)
    fig.savefig(out / "online_pareto.svg", metadata={"Date": None})
    plt.close(fig)
    figures.append(("Goodput and Decode tail", "online_pareto.svg"))
    if not comparable:
        return figures

    metrics = ("TTFT_ms", "TPOT_ms", "ITL_ms", "joint_goodput_req_s", "decode_qualified_goodput_req_s")
    labels = ("TTFT p95", "TPOT p95", "ITL p99", "Joint goodput", "Decode goodput")
    fig, axes = plt.subplots(len(comparable), 1, figsize=(8.5, 3.1 * len(comparable)), squeeze=False, layout="constrained")
    for ax, pair in zip(axes.flat, comparable, strict=True):
        deltas = pair["candidate_over_baseline_relative_change"]
        points = [(label, deltas[key] * 100, key) for label, key in zip(labels, metrics, strict=True)
                  if deltas[key] is not None]
        ax.barh([label for label, _, _ in points], [v for _, v, _ in points],
                color=["#bf5426" if key in ("TPOT_ms", "ITL_ms") and v > 5 else "#1e5f88" for _, v, key in points])
        ax.axvline(0, color="#3c3c3c", linewidth=.8)
        ax.axvline(5, color="#bf5426", linestyle="--", linewidth=.9)
        ax.set(xlabel="Change from the paired Native arm (%)",
               title=f"{pair['candidate'].split('/')[-1]} · strict Decode {pair['strict_decode_verdict']}")
    fig.savefig(out / "online_native_paired.svg", metadata={"Date": None})
    plt.close(fig)
    return figures + [("Paired metric changes", "online_native_paired.svg")]
