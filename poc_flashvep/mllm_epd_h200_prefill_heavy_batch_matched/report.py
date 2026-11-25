"""Render report Markdown from completed H200 records. The GPU launcher remains separate."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from . import analysis as source

PREFIX = "mllm_epd_h200_"
CAPACITY = (
    ("encoder", False, "원래 E"), ("prefill", False, "Embedding-ready P"),
    ("decode", True, "NIXL D Batch1"), ("decode", False, "NIXL D Unrestricted"),
)
BASELINE = ("native-strong-free", "native-common-free", "static10-free")
BATCH = ("native-common-free", "native-common-B1", "static10-B1", "idle10-B1")
RATIOS = (5, 10, 15, 20)


def value(x, decimals=2):
    if x is None or x == "":
        return "—"
    if isinstance(x, bool):
        return "통과" if x else "실패"
    if isinstance(x, (float, int)):
        return f"{x:,.{decimals}f}"
    return str(x).replace("|", "\\|").replace("\n", " ")


def number(row, field):
    x = row.get(field)
    return float(x) if x not in ("", None) else None


def count(x):
    return "—" if x in ("", None) else str(int(float(x)))


def pct(row, field):
    x = number(row, field)
    return "—" if x is None else f"{x * 100:.2f}%"


def f(row, field, digits=2):
    return value(number(row, field), digits)


def table(headers, rows):
    rows = list(rows)
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(value(cell) for cell in row) + " |" for row in rows),
    ])


def path_rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def safe_json(path):
    try:
        with path.open() as stream:
            return json.load(stream)
    except FileNotFoundError:
        return None


def record_name(row):
    return f"`{row['start']}/{row['cohort']}`"


def arm_contract(runs, row):
    run = runs / row["start"]
    config = safe_json(run / "configs" / f"{row['cohort']}.json")
    env = safe_json(run / "state/environment.json")
    return (config["arrival_SHA256"], config["workload"]["manifest_SHA256"],
            env["receipt_SHA256"], env["runtime"]["vllm_wheel_sha256"],
            tuple(sorted(env["authorized_uuid_by_gpu"].items())))


def summary(row):
    return (row["method"], f(row, "QPS", 1), row["output_mode"], record_name(row),
            f(row, "TTFT_mean_ms"), f(row, "TTFT_p50_ms"), f(row, "TTFT_p95_ms"),
            f(row, "ITL_mean_ms"), f(row, "ITL_p99_ms"), f(row, "TPOT_p95_ms"),
            count(row.get("total_output_tokens")), row["evidence"])


def slos(row):
    return (row["method"], f(row, "QPS", 1), row["output_mode"],
            f(row, "total_input_output_tokens_per_second"), f(row, "raw_output_tokens_per_second"),
            f(row, "completed_requests_per_second"), f(row, "SLO_goodput"), pct(row, "TTFT_PASS"),
            pct(row, "D_absolute_SLO_attainment"), pct(row, "JOINT_PASS"))


def render(runs, docs, tables, collection):
    all_rows = tables["cohorts"]
    baseline = [r for r in all_rows if r["start"].startswith("a_")]
    matched = [r for r in all_rows if r["start"].startswith("b_")]
    native_screen = [r for r in all_rows if r["start"].startswith("c_")]
    forward = [r for r in all_rows if not r["start"].startswith(("a_", "b_"))]
    static = [r for r in forward if r["method"].startswith("static")]
    idle = [r for r in forward if r["method"].startswith("idle")]
    latency_header = ["Method", "QPS", "출력 계약", "Run/Cohort", "TTFT mean ms", "TTFT p50 ms",
                      "TTFT p95 ms", "Mean TBT ms", "ITL p99 ms", "TPOT p95 ms", "출력 토큰", "Evidence"]
    slo_header = ["Method", "QPS", "출력 계약", "입력+출력 tok/s", "출력 tok/s", "완료 req/s",
                  "SLO-goodput tok/s", "TTFT Pass", "Decode Pass", "Joint SLO"]

    def put(title, body):
        (docs / (PREFIX + title + ".md")).write_text(body.rstrip() + "\n")

    def latency(rows, heading="### 요청 지연과 작업량"):
        return heading + "\n\n" + (table(latency_header, (summary(row) for row in rows)) if rows
                                     else "완료된 Cohort가 없어 `NOT_RUN`으로 표시한다.")

    def service(rows):
        return "### 처리량과 SLO\n\n" + (table(slo_header, (slos(row) for row in rows)) if rows
                                           else "완료된 Cohort가 없어 `NOT_RUN`으로 표시한다.")

    failures = tables["excluded"]
    def disposition(run_prefix):
        rows = [r for r in failures if r["start"].startswith(run_prefix)]
        if not rows:
            return ""
        return "\n\n" + table(("Run", "Cohort", "Evidence", "원인"),
                                ((r["start"], r["cohort"] or "초기화", r["evidence"], r["reason"]) for r in rows))

    def method_rows(rows, required):
        return [r for r in rows if r["method"] in required]

    def paired_itl(rows):
        by_method = {r["method"]: r for r in rows if r["observation"] == "batch_observed"}
        names = ("native-common-free", "native-common-B1", "static10-B1", "idle10-B1")
        if any(n not in by_method for n in names[:3]):
            return "Unrestricted NativeCommon, NativeCommon-B1, Static10-B1의 매칭 완료 후 격차를 계산한다."
        group = [by_method[n] for n in names[:3]]
        if len({(r["QPS"], r["output_mode"], r["seed"], arm_contract(runs, r)) for r in group}) != 1:
            return "출력·도착 Trace·Manifest·Runtime·GPU가 일치하는 세 Arm이 없어 ITL 효과를 판정할 수 없다."
        free, b1, static_itl = (float(r["ITL_p99_ms"]) for r in group)
        free_gap, b1_gap = free-static_itl, b1-static_itl
        return (f"NativeCommon Unrestricted에서 NativeCommon-B1으로 바꾸자 ITL p99가 {free:.3f}→{b1:.3f}ms였다. "
                f"Static10-B1의 {static_itl:.3f}ms와 비교한 절대 격차는 {abs(free_gap):.3f}→{abs(b1_gap):.3f}ms였다. "
                f"부호 있는 Native−Static 차이는 {free_gap:+.3f}→{b1_gap:+.3f}ms였다. "
                "이는 Counter를 켠 독립 N1의 방향성 근거다. 인과 기여율이나 적격 C3 우위를 확정하지 않는다.")

    observer = []
    for path in sorted(runs.glob("b_*/analysis/*_batch_qualification.json")):
        gate = safe_json(path)
        if not gate:
            continue
        comparisons = gate["metrics"]
        bad = ", ".join(key for key, item in comparisons.items()
                        if item.get("clean_relative_spread", 0) > gate["threshold"]
                        or item.get("observed_relative_deviation", 0) > gate["threshold"])
        observer.append((path.parent.parent.name, gate["passed"],
                         f"{max((x.get('clean_relative_spread',0) for x in comparisons.values()), default=0)*100:.2f}%",
                         f"{max((x.get('observed_relative_deviation',0) for x in comparisons.values()), default=0)*100:.2f}%",
                         bad or "—"))
    observer_table = table(("Run", "5% CPC Gate", "최대 Clean Spread", "최대 Counter Deviation",
                            "실패 Metric"), observer)
    regressions = []
    for path in sorted(runs.glob("b_*/analysis/regression.json")):
        run = path.parent.parent.name
        d = safe_json(path)
        if not d:
            continue
        for phase, result in d.get("result", {}).items():
            if isinstance(result, list):
                regressions.append(("NativeCommon-B1", run, phase,
                                    f"{sum(bool(row['passed']) for row in result)}/{len(result)}",
                                    f"{sum(bool(row['strict_greedy_tokens']) for row in result)}/{len(result)}"))
            elif isinstance(result, dict):
                token_receipt = safe_json(path.with_name(f"irteam_token10_{phase}_summary.json")) or {}
                comparisons = token_receipt.get("full_prompt_comparisons", [])
                strict = (f"{sum(bool(row['strict_tokens']) for row in comparisons)}/{len(comparisons)}"
                          if comparisons else "—")
                regressions.append(("Static10-B1", run, phase,
                                    f"{result.get('requests', '—')}건 {value(result.get('passed'))}", strict))
    regression_table = table(("Arm","Run","검사","상태 Gate","Strict Greedy"),regressions)
    counter_rows = []
    counter_valid = []
    for row in method_rows(matched, BATCH):
        run, tag = runs / row["start"], row["cohort"]
        receipt = safe_json(run / "analysis" / f"{tag}_batch.json")
        if not receipt:
            continue
        receipt, = receipt
        histogram = safe_json(Path(receipt["folder"]) / "histograms.json")
        if not histogram:
            continue
        maxima = receipt.get("decode_maxima", {})
        prepared = [x for x in histogram.get("prepared_decode_padding_histogram", [])
                    if int(x["decode_requests"]) > 0]
        calls = sum(int(x["calls"]) for x in prepared)
        gen_modes = Counter()
        for item in prepared:
            gen_modes[int(item["decode_requests"])] += int(item["calls"])
        proxy_max = pending_max = None
        proxy_path = run / "raw/proxy_epd.jsonl"
        if proxy_path.exists():
            items = [item for item in source.jsonl(proxy_path, appending=True)
                     if item.get("runtime_request_id", "").startswith(tag + "_")]
            intervals = [(item["D_submit"], item["D_http_complete"]) for item in items
                         if "D_submit" in item and "D_http_complete" in item]
            if len(intervals) == 100:
                events = sorted((t, delta) for a,b in intervals for t,delta in ((a,1),(b,-1)))
                active = proxy_max = 0
                for _, delta in events:
                    active += delta
                    proxy_max = max(proxy_max, active)
            snapshots = [item["D_at_P_ready"] for item in items if "D_at_P_ready" in item]
            if len(snapshots) == 100:
                pending_max = max(item["pending_admissions"] for item in snapshots)
        checked = bool(histogram.get("valid")) and calls > 0
        counter_valid.append((row["method"], checked, maxima.get("prepared_decode_requests")))
        counter_rows.append((row["method"], record_name(row), checked,
                             count(maxima.get("scheduled_requests")),
                             count(maxima.get("scheduler_generation_phase_requests")),
                             count(maxima.get("prepared_decode_requests")),
                             count(maxima.get("prepared_query_rows")),
                             count(maxima.get("prepared_query_rows_after_padding")),
                             ", ".join(f"{k}×{v:,}" for k,v in sorted(gen_modes.items())) or "—",
                             count(proxy_max), count(pending_max)))
    batch_table = table(
        ("Method", "Run/Cohort", "Counter 자체 검증", "Scheduled 전체 최대",
         "Scheduler Gen 최대", "Prepared Decode 최대", "Query Rows 최대", "Padded Rows 최대",
         "Decode Requests×Calls", "D HTTP 동시 최대", "P-ready D Pending 최대"), counter_rows)
    batch_status = ("TRACE_ANALYSIS" if observer and all(p[1] for p in observer)
                    else "DIAGNOSTIC" if counter_rows else "NOT_MEASURED")
    coverage_table = table(("통제 Arm", "측정"), [
        (name, "Actual Batch Counter 완료" if any(arm == name for arm, _, _ in counter_valid)
         else "완료된 온라인 Cohort만 있음" if any(row["method"] == name for row in matched)
         else "NOT_RUN")
        for name in ("native-strong-B1", "native-common-B1", "static10-B1", "idle10-B1", "native-common-free")
    ])
    put("decode_batch_matched_control", f"""# H200 Decode Batch1 통제

Actual Batch의 현재 증거 수준은 `{batch_status}`다. Native-B1과 Token-B1은 동일 D FIFO Admission으로 한 번에 한 D 요청만 허용한다. 전체 Client, E, P는 열린 루프와 연속 Scheduling을 유지한다. Native Unrestricted는 별도 실용 기준이다. NIXL과 엔진 KV42GiB는 동일하게 유지하며, 하나의 요청에 대한 Suffix Prefill은 여러 Query Rows를 사용할 수 있다.

{coverage_table}

{batch_table}

Scheduled와 Prepared는 다른 계층의 관측이다. Padding은 실제 CUDA Graph 입력의 행 수이며 Active Decode Request Count로 세지 않는다. D HTTP 동시 최대는 완결된 HTTP 구간 교집합으로, Pending 최대는 P-ready Snapshot으로 계산했다. Pending Snapshot은 관측 가능한 하한이다. Counter 구간의 Geometry와 Client 지표는 원시 기록이 존재하는 Cohort에 한정한다. 같은 프로세스의 Clean/Counter/Clean은 독립 N3에 해당하지 않는다.

## 정합성 회귀 및 계측 검증

{regression_table if regressions else "완료된 Batch1 회귀 Receipt가 아직 없다."}

{observer_table if observer else "완료된 Counter Clean/Observed/Clean Gate Receipt가 아직 없다."}

{latency(matched)}

{service(matched)}

{paired_itl(matched)}
{disposition('b_')}

원시 근거는 각 Run의 `analysis/*_batch.json`, 그 Receipt가 가리키는 `histograms.json`, `raw/proxy_epd.jsonl`에 있다. `actual_d_batch.csv`와 `cohorts.csv`는 재집계 자료다.
""")
    capacity_receipts = []
    for p in sorted(runs.glob("*/analysis/capacity_*.json")):
        data = safe_json(p) or {}
        if data.get("evidence") == "ISOLATED_GPU_MEASUREMENT":
            capacity_receipts.append((p, data))
    capacity_rows = []
    capacity_geometry = []
    for stage, batch_one, label in CAPACITY:
        selections = [(p, j) for p,j in capacity_receipts if j["stage"] == stage
                      and bool(j["batch_one"]) == batch_one]
        if not selections:
            capacity_rows.append((label, "—", "—", "—", "—", "—", "—", "NOT_RUN"))
        for p, j in selections:
            items = j.get("cohorts", [])
            fields = {"encoder": "maximum_actual_encoder_image_batch",
                      "prefill": "maximum_actual_prefill_request_batch",
                      "decode": "maximum_actual_prepared_decode_requests"}
            maximum = [x[fields[stage]] for x in items if x.get(fields[stage]) is not None]
            capacity_rows.append((label, p.parent.parent.name, str(j["requests"]),
                                  str(max(maximum)) if maximum else "—",
                                  value(j["requests_per_second"]), value(j["tokens_per_second"]),
                                  value(j.get("actual_visual_embedding_rows_per_second")),
                                  "ISOLATED_GPU_MEASUREMENT"))
            if stage == "encoder":
                images = sum(row["images_per_second"] * row["duration_seconds"] for row in items)
                capacity_geometry.append((label, "실제 E 이미지/s", value(images / j["seconds"])))
            if stage == "prefill":
                profile = [value(x["P_kernel_busy_fraction"] * 100) + "%" for x in items
                           if "P_kernel_busy_fraction" in x]
                capacity_geometry.append((label, "격리 P Kernel Busy (진단)", ", ".join(profile) or "NOT_MEASURED"))
            if stage == "decode":
                tag = p.stem
                raw = p.parent.parent / "raw" / tag
                decoded = [row for path in raw.glob("*_rows.json") for row in (safe_json(path) or [])]
                itl = [sample for row in decoded for sample in row["ITL_ms"]]
                tpot = [row["TPOT_ms"] for row in decoded]
                transferred = [v["kv_transfer_params"]["remote_num_tokens"]
                               for path in raw.glob("*_prepared.json")
                               for v in (safe_json(path) or {}).get("prefill_export", [])]
                capacity_geometry.extend((
                    (label, "Client ITL p99 / TPOT p95 (ms)",
                     f"{value(source.percentile(itl, .99))} / {value(source.percentile(tpot, .95))}"),
                    (label, "실제 P KV Export Tokens p50 / p95",
                     f"{value(source.percentile(transferred, .50))} / {value(source.percentile(transferred, .95))}"),
                ))
    capacity_table = table(
        ("Stage", "Run", "완료 요청", "실제 최대 Batch", "req/s", "tok/s", "실제 E Rows/s", "Evidence"), capacity_rows)
    proximity = tables["service_proximity"]
    closest = [r for r in proximity if r["role"] == "prefill"]
    proximity_table = table(("Run/Cohort", "QPS", "단위", "Online P", "Isolated P", "Proximity", "Evidence"),
                            [(record_name(r), f(r, "QPS", 1), r["units"], f(r, "online_rate"), f(r, "isolated_rate"),
                              pct(r, "rate_ratio"), r["evidence"]) for r in closest])
    geometry_table = table(("Stage", "관측", "값"), capacity_geometry)
    put("independent_stage_capacity", f"""# H200 독립 Stage Capacity

같은 Main100 원문과 원래 Encode·P·D Engine을 사용한다. E는 실제 원본 이미지의 동시 Encoding을 사용한다. P에는 앞서 원래 E가 준비한 Embedding을 제공한다. P 측정은 D Backpressure가 없는 로컬 KV 반환 경계를 사용한다. D는 원래 P의 NIXL KV와 동결 5,269 Output Token 분포를 사용하며 E/P 준비 구간을 분모에서 제외한다. D-only B1과 Unrestricted는 따로 측정한다.

{capacity_table}

E와 P의 최대 Batch는 각각 실제 Encoder 출력 이미지 수와 Prepared Prefill 요청 수다. D는 Actual Prepared Decode 요청 수다. E의 `tok/s`는 Manifest Visual Row 기준이며 실제 출력을 따로 기록한 경우 `실제 E Rows/s`에서 확인한다. P는 Input Token, D는 Output Token이다. Cohort 처리 시간은 각 Stage 첫 제출부터 마지막 완료까지이며 Cohort의 외부 준비 및 Drain 구간 정의를 Raw에 남긴다. Counter 오버헤드와 Steady State는 별개의 Gate다.

{geometry_table if capacity_geometry else "Stage Geometry 상세는 현재 `NOT_MEASURED`다."}

### Online / Isolated P 비교

{proximity_table if closest else "독립 P와 비교 가능한 Online Series가 아직 없어 `NOT_MEASURED`다."}

Stage 분리는 병목 진단이다. P의 로컬 KV 반환과 Online KV Export·NIXL 수명, Actual Batch와 Attention Geometry 차이는 같은 조건이 확인될 때까지 보존한다. `DIAGNOSTIC` Proximity를 포화 증명으로 단독 사용하지 않는다.
{disposition('c_capacity')}

원시는 각 Run의 `analysis/capacity_*.json`, `raw/capacity_*` 및 `stage_capacity.csv`, `service_proximity.csv`에 있다.
""")
    qualifications = []
    for p in sorted(runs.glob("*/analysis/*_profile_gate.json")):
        gate = safe_json(p)
        if not gate:
            continue
        trace_path = p.with_name(p.name.removesuffix("_profile_gate.json") + "_trace.json")
        trace = safe_json(trace_path) or {}
        metric = [(name, x) for name, x in gate["metrics"].items() if not x["passed"]]
        bad = "; ".join(f"{name} (spread {pct(x, 'clean_relative_spread')}, deviation {pct(x, 'profile_relative_deviation')})"
                        for name, x in metric)
        qualifications.append((p, gate, trace, bad or "—"))

    discoveries = []
    for p in sorted(runs.glob("**/*discovery*.json")):
        if p.name == "download_plan.json" or "raw" in p.parts:
            continue
        data = safe_json(p)
        if isinstance(data, dict) and str(data.get("status", "")).startswith("P_HEAVY_D_IDLE_"):
            discoveries.append((p, data))
    statuses = [j["status"] for _, j in discoveries
                if j.get("batch_one", True) and j.get("main100_cycles", 1) == 1]
    if "P_HEAVY_D_IDLE_CONFIRMED" in statuses:
        mot = "P_HEAVY_D_IDLE_CONFIRMED"
    elif "P_HEAVY_D_IDLE_NOT_CONFIRMED" in statuses:
        mot = "P_HEAVY_D_IDLE_NOT_CONFIRMED"
    else:
        mot = "P_HEAVY_D_IDLE_NOT_MEASURED"
    busy = {(r["start"], r["cohort"], r["role"]): pct(r, "busy_fraction") for r in tables["kernel_busy"]}
    profile_points = [r for r in all_rows if r["observation"] == "profile"]
    qps = [r for r in native_screen if r["method"].startswith("native") and r["output_mode"] == "Natural EOS"]
    online_table = table(("Method", "QPS", "Run/Cohort", "E Busy", "P Busy", "D Busy", "P req/s",
                          "P tok/s", "P HTTP 존재", "D HTTP/대기 없음", "TTFT p95 ms", "ITL p99 ms",
                          "TPOT p95 ms", "Evidence"),
                         [(r["method"], f(r, "QPS", 1), record_name(r),
                           *(busy.get((r["start"], r["cohort"], role), "—")
                             for role in ("encoder", "prefill", "decode")),
                           f(r, "P_reqs_per_central_second"), f(r, "P_input_tokens_per_central_second"),
                           pct(r, "P HTTP_central_nonempty_fraction"),
                           pct(r, "No active D HTTP or P-ready wait_central_fraction"), f(r, "TTFT_p95_ms"),
                           f(r, "ITL_p99_ms"), f(r, "TPOT_p95_ms"), r["evidence"])
                          for r in sorted(qps + [r for r in profile_points if r not in qps],
                                          key=lambda r: (r["method"], number(r, "QPS"), r["start"], r["cohort"]))])
    gate_table = table(("Run", "Gate", "Collector", "실패한 Metric과 간섭"),
                       [(p.parent.parent.name, gate["evidence"], gate.get("collector_passed", False), bad)
                        for p,gate,trace,bad in qualifications])
    reasons = [
        f"- `{path.name}`은 `{data['status']}`다. 미통과 적용 항목은 "
        + ", ".join(k for k,v in data.get("checks", {}).items() if not v) + "이다."
        for path,data in discoveries if any(not v for v in data.get("checks", {}).values())
    ]
    discovery_table = table(("Receipt", "D 정책", "Main100 반복", "판정", "P Busy", "P req/s Proximity", "P tok/s Proximity",
                             "P 최근 Waiting", "D Decode SLO", "D Quiet ≥160ms의 P Busy Coverage",
                             "D Quiet ≥225ms의 P Busy Coverage"), [
        (path.name, "Batch1" if data.get("batch_one", True) else "Unrestricted", data.get("main100_cycles", 1),
         data["status"], pct(data, "p_busy"),
         pct(data.get("p_service_proximity", {}), "requests_per_second"),
         pct(data.get("p_service_proximity", {}), "tokens_per_second"),
         pct(data.get("p_scheduler", {}), "window_fraction_recent_sample_waiting"),
         pct(data, "d_decode_attainment"),
         pct(data.get("robust_window_160_ms", {}), "p_busy_overlap_fraction"),
         pct(data.get("robust_window_225_ms", {}), "p_busy_overlap_fraction"))
        for path, data in discoveries
    ])
    put("prefill_heavy_discovery", f"""# H200 Prefill-heavy 탐색과 계측

현재 Motivation 판정은 `{mot}`다. Capacity Knee 탐색은 실제 E/P/D Online Main100에서 수행하며, 적격 P Kernel Busy·비교 가능한 독립 P Capacity·실제 P Backlog·D Quiet 구간·Decode SLO를 함께 확인한다. E가 바쁜 구간은 포함한다.

## A. QPS2 Baseline 복구

{latency(method_rows(baseline, BASELINE))}

{service(method_rows(baseline, BASELINE))}

Setup Correctness와 이번 실제 open-loop Online 구간은 별개다. H100에서 재현한 수치와 직접 알고리즘 Speedup을 계산하지 않는다. `MEASURED_ONLINE`은 개별 100요청 Cohort를 뜻한다. H200 NativeStrong/Main100은 별도 Workload로 유지한다.
{disposition('a_')}

## C–E. QPS 탐색 및 계측

{online_table if qps or profile_points else "Native QPS 탐색과 Kernel Profile은 아직 `NOT_RUN`이다."}

{service(qps)}

P HTTP는 Host의 진행 중 요청 구간이며 P Backlog나 Kernel Busy 범위를 과대 대표할 수 있다. D HTTP와 P-ready 대기가 없는 구간도 Suffix 안전성의 보수적 Proxy다. Kernel·Scheduler 관측이 적격인 Profile만 P-heavy 결론을 지지한다. Capture는 p20–p80 예정 Arrival과 실제 공통 ROI의 교집합이다.

{gate_table if qualifications else "Clean/Profile/Clean 판정 기록이 아직 없다."}

{discovery_table if discoveries else "Motivation Gate의 수치 평가 기록이 아직 없다."}

{chr(10).join(reasons) if reasons else "P Compute 병목, E 공급 제한, KV Admission, D Decode 또는 HTTP 대기 중 단일 원인을 현재 증거로 확정하지 않는다."}
{disposition('c_')}

`cohorts.csv`, `kernel_busy.csv`, `kernel_overlap.csv`, 실제 Trace와 `analysis/*_profile_gate.json`이 재현 자료다.
""")
    prefix_rows = [r for r in all_rows if number(r, "P_ready_D_observations")]
    prefix_table = table(("Run/Cohort", "Method", "QPS", "관측 건", "D HTTP 비활성",
                          "D 비활성 및 대기 없음", "D 대기 p50 ms", "p95 ms", "p99 ms", "실제 Split 건"),
                         [(record_name(r),r["method"],f(r,"QPS",1),count(r.get("P_ready_D_observations")),
                           pct(r,"P_ready_D_HTTP_inactive_fraction"),pct(r,"P_ready_D_HTTP_inactive_no_pending_fraction"),
                           f(r,"D_coordinator_wait_p50_ms"),f(r,"D_coordinator_wait_p95_ms"),
                           f(r,"D_coordinator_wait_p99_ms"),count(r.get("split_requests")))
                          for r in prefix_rows])
    d_windows = []
    window_rows = tables["decode_idle_windows"]
    by_window = {}
    for row in window_rows:
        key = (row["start"], row["cohort"], row["series"])
        by_window.setdefault(key, []).append(float(row["duration_seconds"]) * 1000)
    for (run, tag, series), samples in by_window.items():
        if len(samples) == 0:
            continue
        d_windows.append((run, tag, series, str(len(samples)), value(source.percentile(samples,.50)),
                          value(source.percentile(samples,.95)),
                          f"{sum(x for x in samples if x >= 160)/sum(samples)*100:.2f}%",
                          f"{sum(x for x in samples if x >= 225)/sum(samples)*100:.2f}%"))
    d_table = table(("Run","Cohort","Idle 정의","Window 수","p50 ms","p95 ms","160ms 이상 Duration","225ms 이상 Duration"), d_windows)
    overlap = tables["kernel_overlap"]
    overlap_table = table(("Run/Cohort", "교집합", "중앙 ROI 대비", "P Busy 대비", "Evidence"), [
        (record_name(row), row["condition"], pct(row, "fraction_all"), pct(row, "fraction_P_busy"), row["evidence"])
        for row in overlap
    ])
    put("decode_idle_prefix_ready_analysis", f"""# H200 Decode Idle과 Prefix-ready 정렬

`D_at_P_ready`는 P 응답 직후 D의 Active HTTP·Pending Coordinator 요청을 기록한다. Native 관측은 Full Prefill 완료 시점이다. 실제 Static/Idle Token Split에 적용된 행에서만 이 값이 Suffix 이전 Prefix 완료의 실제 관측이다. HTTP는 Worker의 실제 D Prepared Decode와 물리적 CUDA Kernel Idle을 보수적으로 근사한다.

{prefix_table if prefix_rows else "Prefix-ready D 상태 자료는 `NOT_MEASURED`다."}

## Idle Window의 길이

{d_table if d_windows else "확인된 Idle Window가 없다."}

{overlap_table if overlap else "P와 D의 Kernel 교집합은 `NOT_MEASURED`다."}

160ms와 225ms는 기존 H100 진단 비용에 근거해 사전 선택한 검토 길이다. Duration 비율 분모는 시리즈별 Idle Duration이다. P Busy와의 실제 교집합은 적격 CUPTI Trace와 공통 ROI에서만 확인한다. Future Idle Window를 알 수 있다는 가정을 둔 Fit은 `OFFLINE_ORACLE`이다.

{chr(10).join(f"- `{p.name}`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 {pct(d.get('robust_window_160_ms',dict()), 'p_busy_overlap_fraction')}다." for p,d in discoveries) if discoveries else "동시 P Busy·D Quiet 및 실제 KV 여유는 적격 판정 기록이 생길 때까지 `NOT_MEASURED`다."}

재현 자료는 `decode_idle_windows.csv`, `kernel_overlap.csv`, `requests.csv`와 각 이미지·KV Owner Receipt다.
""")
    c3_names = {r["method"] for r in static}
    ratio_rows = [(f"Static{i}-B1", "완료 Cohort 확인" if f"static{i}-B1" in c3_names else "NOT_RUN")
                  for i in RATIOS]
    put("static_token_split_online_serving", f"""# H200 Static Token Split Online Serving

Phase F의 P-heavy 확인 이후 후보 실험은 `{('MEASURED_ONLINE' if static else 'NOT_RUN')}`이다. Phase B의 QPS2 Batch1 대조군은 Admission/Geometry 진단 근거다. E는 원래 Vision을 유지하고 P contiguous Prefix와 D contiguous Suffix를 사용한다. D Active Decode가 없어야 Suffix 제출을 허용한다.

{table(("후보", "Phase F 진행"), ratio_rows)}

{latency(static)}

{service(static)}

## Phase B QPS2 Equal-output 통제

{latency([r for r in matched if r["method"] in ("native-common-B1", "static10-B1")])}

{paired_itl(matched)}

Static 성능의 기여는 Same Batch1 Native와 Paired Independent Starts에서 평가한다. Unrestricted Native 대비 결과는 실용 기준으로 남긴다. H100/H200, Natural/Replay, Batch Counter/ Clean의 측정군을 따로 판단한다.
{disposition('f_')}
""")
    idle_cal = [r for r in matched if r["method"] == "idle10-B1"]
    idle_table = table(("Run/Cohort", "Step", "Split 실행 건", "원래 Native 실행 건", "이전 Tokens",
                        "전체 Offload", "D Wait p95 ms", "D at P-ready no pending"),
                       [(record_name(r), "Phase B" if r in idle_cal else "Phase G", count(r.get("split_requests")),
                         str(100-int(n)) if (n:=number(r,"split_requests")) is not None else "—",
                         count(r.get("suffix_tokens")), pct(r,"actual_offload_fraction"),
                         f(r,"D_coordinator_wait_p95_ms"), pct(r,"P_ready_D_HTTP_inactive_no_pending_fraction"))
                        for r in idle_cal + idle])
    put("idle_aware_token_split_online_serving", f"""# H200 Idle-aware Token Split Online Serving

Phase G는 `{('MEASURED_ONLINE' if idle else 'NOT_RUN')}`이다. 기존 Idle10 정책에서는 계획 시 P HTTP가 진행 중이고 D HTTP가 비었을 때만 Split한다. 조건이 충족되지 않으면 Native 경로를 사용하며 Prefix 완료 후 보호 조건을 다시 확인한다.

{idle_table if idle_cal or idle else "실행된 Idle10 Cohort가 없다."}

{latency(idle_cal + idle)}

{service(idle_cal + idle)}

낮은 Split 실행률은 전체 효과 판단에 포함한다. Phase B의 Counter QPS2 실험만으로 사전 확인된 P-heavy에서 Static10 대비 추가 개선을 결론 내리지 않는다. 개선 정책과 미래 Idle 예측은 확인된 Prefill-heavy의 Static 결과 이후에만 평가한다.
{disposition('g_')}
""")
    selected_paths = [r for r in tables["critical_path"] if r["start"].startswith(("a_","b_","f_","g_"))]
    by_path = {}
    for r in selected_paths:
        by_path.setdefault((r["start"],r["cohort"],r["method"]),{})[r["stage"]] = float(r["mean_ms"])
    stages = source.CRITICAL_PATH
    path_table = table(("Run/Cohort", "Method", *stages, "합계 ms"),
                       [(f"`{run}/{tag}`", method, *(value(group[s]) for s in stages), value(sum(group.values())))
                        for (run,tag,method),group in by_path.items() if len(group) == len(stages)])
    ddelta = by_path.get(next((k for k in by_path if k[2] == "static10-B1" and k[0].startswith("b_")), None))
    ndelta = by_path.get(next((k for k in by_path if k[2] == "native-common-B1" and k[0].startswith("b_")), None))
    path_change = ("TTFT 임계경로가 짝지어진 Run에서 확보되면 구간 차이를 기록한다." if ddelta is None or ndelta is None else
                   f"QPS2 Phase B의 Static10-B1−NativeCommon-B1에서 P HTTP {ddelta['P HTTP']-ndelta['P HTTP']:+.3f}ms, "
                   f"P-ready→D-submit {ddelta['P ready to D submit']-ndelta['P ready to D submit']:+.3f}ms, "
                   f"전체 평균 TTFT {sum(ddelta.values())-sum(ndelta.values()):+.3f}ms였다. "
                   "같은 출력 계약의 Host 경계 진단이며 순수 GPU 연산 비용은 별개다.")
    put("token_split_critical_path", f"""# H200 Token Split의 TTFT 및 ITL 임계경로

요청별 Scheduled Arrival부터 First Client Token까지 겹치지 않는 Host 경계를 합산한다. 평균의 가법 분해와 TTFT p95의 분포 변화는 분리한다. 아래 모든 경계의 단위는 ms다. Host HTTP 구간에는 스케줄링, 전송, GPU 계산과 응답이 포함된다.

{path_table if by_path else "완전한 동일 요청의 임계경로가 없어 `NOT_MEASURED`다."}

{path_change}

D Suffix의 GPU-exclusive 시간, KV Import 내부와 D First Worker Output은 실제 해당 Trace의 구분이 가능한 범위까지만 기록한다. D Submit→First Client의 Host 경계에서 Suffix 전체 비용을 역산하지 않는다.

{paired_itl(matched)}

[H100 Critical Path](mllm_epd_protected_token_critical_path_v1.md)의 Static10 Replay 수치는 P HTTP −305.329ms, D Coordinator +971.956ms, 평균 TTFT +933.063ms였다. Hardware/Runtime이 바뀐 H200의 차이에 이 값을 직접 차감하지 않는다.
""")
    actual_status = "Actual Batch Counter가 아직 없다."
    if counter_valid:
        actual_status = "Prepared Decode 최대값 " + ", ".join(
            f"{arm} {count(maximum)}" + ("" if valid else " (Counter 무효)")
            for arm,valid,maximum in counter_valid) + "을 관측했다. 같은 QPS2 Equal-output Counter Cohort의 근거다."
    actual_any = next((r for r in prefix_rows if r["method"] == "static10-B1"), None)
    ready = ("실제 Token Split의 Prefix-ready 자료는 아직 없다." if not actual_any else
             f"Phase B Static10-B1의 {count(actual_any['P_ready_D_observations'])}건은 D HTTP 비활성 "
             f"{pct(actual_any,'P_ready_D_HTTP_inactive_fraction')}, 비활성·대기 없음 "
             f"{pct(actual_any,'P_ready_D_HTTP_inactive_no_pending_fraction')}, Coordinator Wait p95 "
             f"{f(actual_any,'D_coordinator_wait_p95_ms')}ms였다. 적격 P-heavy 결과는 별도다.")
    diagnostic = ("P Compute 여부를 분리할 적격 P Busy, 독립 P Capacity 및 실제 Backlog 교차 증거가 부족하다."
                  if not discoveries else " ".join(reasons) or "Gate의 모든 정량 항목을 해당 Discovery Receipt에서 확인했다.")
    stage_modes = {(record["stage"], record.get("batch_one", False)) for _, record in capacity_receipts}
    stage_label = "측정 완료 " + str(len(stage_modes)) + "/4" if stage_modes else "NOT_RUN"
    conclusion = ("HOLD" if not static or mot != "P_HEAVY_D_IDLE_CONFIRMED" else
                  "HOLD — 적격 Paired CI와 전체 품질의 별도 최종 검증 필요")
    final_questions = [
        ("Q1. H200 환경과 Dataset을 재구축했는가?", "예. 고정 모델·Media·NPZ·Manifest 해시와 H200 Runtime Fingerprint를 검증했다. Native N8 상태 8/8, Greedy 6/8, Token10 두 기준 8/8이다. Wheel SHA와 Driver/Hardware는 H100과 다르다."),
        ("Q2. Actual Decode Batch를 1로 통일했는가?", actual_status),
        ("Q3. Batch1 이후 Static10과 ITL 격차는?", paired_itl(matched)),
        ("Q4. P-heavy와 D Idle이 동시 발생하는 QPS를 찾았는가?", f"`{mot}`. 핵심 Gate는 적격 Trace, 실제 P Backlog 및 독립 Capacity다."),
        ("Q5. P-heavy의 근거가 부족한 이유는?", diagnostic),
        ("Q6. 실제 Prefix-ready에서 D가 얼마나 자주 비어 있었는가?", ready),
        ("Q7. 확인된 P-heavy에서 Static Split이 TTFT를 개선했는가?", "Phase F의 완료 Cohort는 " + str(len(static)) + "개다. Batch1과 Unrestricted의 Paired 기준을 별도로 평가한다."),
        ("Q8. ITL/TPOT가 유지됐는가?", "Phase B 진단의 수치는 표에 있다. 적격 P-heavy에서의 Decode 비열등성과 품질은 Paired CI 및 Absolute 95%로 별도 검사한다."),
        ("Q9. Static10 대비 Idle10의 추가 이득은?", "Phase G의 완료 Cohort는 " + str(len(idle)) + "개다. QPS2 진단의 Split 실행률, 이전 Token, Native 경로 및 D 대기는 별도 보고서에 남겼다."),
        ("Q10. 기존 H100 실패의 원인은?", "Static10에서 P HTTP 절감보다 D Admission 대기가 더 컸다는 경로 변화가 직접 관측됐다. Actual Decode Batch 차이는 관측 간섭 실패 N1에 해당한다. Prefill-heavy와 독립 Stage Capacity는 이전에 미입증이었다."),
        ("Q11. Modality-aware로 확장할 근거가 있는가?", "`HOLD`. 실제로 확인된 P-heavy, Batch1 일치, TTFT 개선과 Decode·품질 보호가 필요하다."),
    ]
    figure_names = sorted(p.stem for p in collection.glob("*.png"))
    figure_line = ("재생성된 Figure는 " + ", ".join(f"`{name}`" for name in figure_names) + "이다."
                   if figure_names else "Figure 재생성은 증거 스냅샷의 집계 명령에 포함한다.")
    put("prefill_heavy_token_split_final_summary", f"""# H200 MLLM EPD Prefill-heavy와 Batch-matched Token Split 종합

연구 판정은 `{conclusion}`다. Motivation 판정은 `{mot}`다. 현재 독립 Stage Capacity 상태는 {stage_label}이다.

{chr(10).join(f"**{q}** {a}\n" for q,a in final_questions)}

## 결과표

{latency(baseline + matched)}

{service(baseline + matched)}

## 재현과 남은 Gate

원본 Setup과 이번 H200 Run은 같은 서버에 보존한다. 핵심 자료는 `h200_runs/<run>/state`, `configs`, `raw`, `analysis`에 있고 각 Report에 Evidence Level을 표시했다. 집계는 `python -m poc_flashvep.mllm_epd_h200_prefill_heavy_batch_matched.analysis --runs <h200_runs> --output <new-output>`으로 재생성한다.

{figure_line}

- [사양서]({PREFIX}prefill_heavy_batch_matched_poc_spec.md)
- [환경 및 초기 정합성]({PREFIX}environment_and_correctness.md)
- [Decode Batch1 통제]({PREFIX}decode_batch_matched_control.md)
- [독립 Stage Capacity]({PREFIX}independent_stage_capacity.md)
- [Prefill-heavy 탐색]({PREFIX}prefill_heavy_discovery.md)
- [Decode Idle과 Prefix-ready]({PREFIX}decode_idle_prefix_ready_analysis.md)
- [Static Token Split]({PREFIX}static_token_split_online_serving.md)
- [Idle-aware Token Split]({PREFIX}idle_aware_token_split_online_serving.md)
- [TTFT 및 ITL 임계경로]({PREFIX}token_split_critical_path.md)
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--docs", required=True, type=Path)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    args.runs = args.runs.resolve()
    if not args.runs.is_dir() or not args.docs.is_dir():
        parser.error("--runs and --docs must identify existing directories")
    if args.analysis is not None and not (args.analysis / "receipt.json").is_file():
        parser.error("--analysis must contain a completed aggregate receipt")
    keys = ("cohorts", "excluded", "service_proximity", "kernel_busy", "kernel_overlap",
            "decode_idle_windows", "critical_path")
    with TemporaryDirectory(prefix="h200-report-") as tmp:
        collection = args.analysis or (Path(tmp) / "data")
        if args.analysis is None:
            source.aggregate(args.runs, collection, figures=False)
        tables = {key:path_rows(collection / (key + ".csv")) for key in keys}
        render(args.runs, args.docs, tables, collection)


if __name__ == "__main__":
    main()
