"""Build an interim record from read-only candidate cards and status evidence."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

# These statements are reviewed synthesis. They are not inferred test results.
# Fields: exactness, prior, boundary, models, fixed, adaptive, P2, P4, unresolved, sources.
NOTES = {
    "C01": (
        "Intended graph-preserving execution, with numerical reordering.",
        "NanoFlow, CDB, generic aliased operator batching, persistent task workers.",
        "Ordinary ready-task pooling has direct generic controls. Later same-token inputs are not ready.",
        "Both model shapes audited. Shared component tests do not execute this candidate.",
        "Source/DAG screen. No candidate request execution.", "No candidate original-gate execution.",
        "Related MLP components only, not worker architecture.", "Related MLP components only, not worker architecture.",
        "Distinct ownership optimum, actual online ready frontier, full state protocol, model numerics, and performance.",
        ["reports/operator_ownership_nearest_audit.md", "reports/persistent_execution_redteam.md", "reports/interpass_cache_capacity_screen.md", "reports/interpass_cache_capacity_redteam.md"]),
    "C02": (
        "Graph-preserving remote MLP. Tested returned component words match local words.",
        "Attention/FFN disaggregation, KernelFlume, local batching, native TP.",
        "Remote full-MLP round trip loses at every tested row count, 1 through 512, for both checkpoints.",
        "Both actual checkpoints, component kernels only.",
        "No full candidate fixed-R4 request execution.", "No full candidate original-gate request execution.",
        "Physical graph-captured MLP component measured.", "Physical graph-captured MLP component measured.",
        "Other legal pooled frontiers remain untested. Component parity is not task-quality evidence.",
        ["reports/cost_and_headroom.md", "analysis/checkpoint_03/operators.csv"]),
    "C03": (
        "Intended exact address/value policy. No model-output parity result.",
        "Enabling Dynamic Computation Ragged KV, WaveFront depth addressing, paged attention.",
        "Terminal aliasing has a direct prior and saves no fixed-R4 terminal copies. Mapping/interface tests do not establish an allocator.",
        "729 synthetic exit histories and 93312 K/V comparisons. Four-row GPU interface fixture, not model outputs.",
        "Distinct computed depth states remain distinct. No live alias runtime.",
        "Synthetic terminal-depth map and interface tests only. Original-gate candidate execution NOT_RUN.",
        "Distributed candidate NOT_RUN.", "Distributed candidate NOT_RUN.",
        "Compact allocator, online publication, model parity, address overhead, and an additional ownership mechanism.",
        ["reports/kv_indirection_redteam.md", "reports/dynamic_runtime_prior_boundary.md"]),
    "C04": (
        "Real-arithmetic valid layout identity. BF16 boundary equivalence not automatic.",
        "TP/SP layout propagation, TSP, collective/norm fusion, generic dataflow.",
        "CPU valid layout matches within 1.78e-15. Constructed separate nonlinear treatment of additive partials fails.",
        "CPU algebra with relevant dimensions, not candidate model execution.", "No candidate request result.",
        "Original-gate candidate NOT_RUN.", "Structural layout only, not distributed candidate execution.",
        "Structural layout only, not distributed candidate execution.",
        "A genuinely removed message, BF16 numerical contract, and net gain over valid generic fusion.",
        ["reports/layout_and_structured_residual_redteam.md", "reports/folded_layout_transfer_audit.md"]),
    "C05": (
        "Provisional correction. Exact only at a sufficient state/KV recovery point.",
        "DEER, ELK/quasi-DEER, SNLP, nonlinear finite-chain correction and recurrent samplers.",
        "Two single-token domain probes per size need four zero-warm-up rounds, or one exact pass plus three rounds, for complete tested recovery.",
        "Both actual checkpoints. Serial one-GPU state/KV diagnostic, not a physical parallel solver.",
        "Full-R4 finite trace tested. Earlier token agreement does not certify KV.",
        "No original adaptive candidate correctness or physical-skip result.",
        "Analytic cost projection only.", "Analytic cost projection only.",
        "Cheaper complete correction, verified KV, charged same-P latency, and free-running quality for approximation.",
        ["reports/parallel_sampling_nearest_audit.md", "reports/cost_and_headroom.md", "outputs/finite_correction_base14_v1/finite_correction.json", "outputs/finite_correction_base26_v1/finite_correction.json"]),
    "C06": (
        "Exact linear distributivity in real arithmetic. BF16 reordering remains separate.",
        "Generic linear distributivity. FlashLoop approximations are a separate comparator.",
        "The exact dense formula retains both base and fresh-delta GEMMs. Small delta norm does not establish cheap exact work.",
        "CPU algebra/BF16 counterexamples and bounded actual activations from both sizes.",
        "No candidate execution.", "No candidate execution.", "Candidate rewrite NOT_RUN.", "Candidate rewrite NOT_RUN.",
        "A proved cheaper residual representation and model numerical/quality evidence.",
        ["reports/layout_and_structured_residual_redteam.md", "reports/shared_weights_selected_computations_audit.md"]),
    "C07": (
        "Conditional verified speculation. Greedy and distribution-preserving guarantees require separate complete protocols.",
        "LoopSpec, WaveFront, Hydragen, PipeSpec/PipeInfer; ordinary TP of identical branch work.",
        "Current replica ownership overlaps generic verified batching and must charge tentative ancestor KV. No physical candidate falsifier completed.",
        "Both sizes are targets. Eager one-pass component has 15 CPU tests in both interpreters. GPU fixture testing PENDING.",
        "Perfect-acceptance ready-frontier component only is planned. No closed-loop speculation.",
        "Original-gate distributed verification UNVALIDATED.",
        "Candidate NOT_RUN. Current component contract does not supply TP2.",
        "Eager TP4/four-row component GPU testing PENDING. TP1/one-row GPU testing PENDING.",
        "FA3 metadata support, exact component parity, service time, four simultaneous TP1 owners, natural frontier, acceptance, rollback/commit, compiled baseline, and task quality.",
        ["reports/branch_ownership_cost_contract.md", "reports/hydragen_c07_transfer_audit.md", "reports/depth_sharded_verifier_prior_audit.md", "reports/c07_one_pass_fixture_contract_v1.md", "artifacts/c07_one_pass_fixture_cpu_validation_v1.json"]),
    "C08": (
        "Lossless byte representation only after exact word reconstruction. BF16 subtraction alone is not lossless.",
        "Generic lossless transport coding and reversible transforms.",
        "Bounded encoded-size savings do not pay measured small-message startup automatically. Unfused transform already adds comparable cost before the entropy codec.",
        "CPU coding of 108 captured activation pairs per checkpoint. GPU primitive is not a full codec.",
        "Captured fixed-decode ratios are descriptive. No compressed candidate request.",
        "No candidate adaptive result.", "Physical raw transport and unfused reversible-transform probe.",
        "Full compressed transport NOT_RUN.",
        "Fused encoder/decoder, exact receiver-base management, real removed communication, and complete critical-path timing.",
        ["reports/cost_and_headroom.md", "artifacts/lossless_base14.json", "artifacts/lossless_base26.json"]),
    "C09": (
        "Exact greedy certificate only with valid finite-precision bounds. No sampling guarantee follows.",
        "FEXIPRO/exact MIPS, branch-and-bound dot products, vocabulary-parallel argmax.",
        "Known final-hidden bound is generic. A close earlier hidden can change the winner in the CPU counterexample.",
        "Synthetic rational/BF16 and source checks. No candidate model or GPU test.",
        "No model head certificate test.", "No original-gate candidate test.", "Candidate NOT_RUN.", "Candidate NOT_RUN.",
        "Useful conservative BF16 bounds, actual head hit rates/costs, and a separate later-hidden enclosure.",
        ["reports/certified_decision_redteam.md", "artifacts/certified_decision_cpu.json"]),
    "C10": (
        "Intended exact rematerialization, with GEMM-shape and KV parity unresolved.",
        "Residual Stream, CacheFlow restoration, generic rematerialization.",
        "Source/cost boundary only. Reprojection and state publication remain charged. No candidate empirical falsifier completed.",
        "Both model geometries considered. No rematerialization model run.", "Candidate NOT_RUN.",
        "Candidate NOT_RUN.", "Candidate NOT_RUN.", "Candidate NOT_RUN.",
        "Real avoided capacity/transfer, complete history reprojection costs, numerical parity, and distinct repeated-operator reuse.",
        ["reports/independent_axes_redteam.md", "reports/residual_input_reassociation_redteam.md"]),
    "C11": (
        "Real-arithmetic coded-linear identity. Native BF16 equivalence unresolved.",
        "Short-Dot and generic coded computation.",
        "Generic prior overlap. No measured native straggler tail or complete coded-path falsifier.",
        "Source/algebra formulation, no coded candidate model run.", "Candidate NOT_RUN.",
        "Candidate NOT_RUN.", "Candidate NOT_RUN.", "Candidate NOT_RUN.",
        "Removable real tail, encoding/decoding and redundant work, numerical contract, and same-resource gain.",
        ["reports/independent_axes_redteam.md"]),
    "C12": (
        "Approximate softmax unless an exact sufficient representation is proved.",
        "Performer features, generic linear attention and context parallelism.",
        "Exact rational rank-N response construction rejects a universal constant-size linear-feature claim. It does not reject every nonlinear or approximate representation.",
        "CPU rank constructions N2 through N8. Separate selected native keys from both checkpoints, not a summary implementation.",
        "Selected key captures only. No summary candidate execution.", "No candidate adaptive result.",
        "Candidate NOT_RUN.", "Candidate NOT_RUN.",
        "Concrete representation, declared approximation class, state size, actual attention error, quality, and physical bottleneck.",
        ["reports/attention_summary_exactness_screen.md", "artifacts/attention_summary_rank_v2.json", "artifacts/key_rank_base14_v1.json", "artifacts/key_rank_base26_v1.json", "reports/cross_depth_pattern_certificate.md"]),
    "C13": (
        "Conditional exact-input continuation with original rounded hazards and private state.",
        "Generic conditional/speculative control execution and asynchronous gate transport.",
        "R2 can overlap R1 transport under minimum R2. Existing delayed consumption is not a same-policy substitute. Synthetic promotion-before-sigmoid changes an exit.",
        "Source contract and synthetic CPU precision check. No overlap implementation.",
        "No adaptive-gate wait exists to remove in fixed R4.", "Baseline opportunity only. Live exits/performance NOT_RUN.",
        "Candidate NOT_RUN.", "Candidate NOT_RUN.",
        "GPU-rounded hazard parity, snapshots/lifetimes, provisional cancellation/publication, and exposed control cost.",
        ["reports/original_gate_overlap_contract.md", "artifacts/original_gate_overlap_contract.json"]),
    "C14": (
        "Graph-preserving causal PP, with numerical reordering still requiring qualification.",
        "TeraPipe, tied-aware generic cyclic PP, Fast-LLM shared buffers, generic KV placement.",
        "Matched generic physical trace uses identical weight owners, aliases, chunk DAG, depth KV, kernels, messages, and schedule. Not only DAG representability.",
        "CPU graph/alias toy and pinned sources. No physical Ouro cyclic-PP model run.",
        "Full-R4 prefill causal chunks are legal. C1 chunks can overlap. No PP timing result.",
        "Prefill remains full R4. Online original-gate PP integration NOT_RUN.",
        "Structural ownership/capacity accounting only.", "Structural ownership/capacity accounting only.",
        "A genuinely non-identical ownership/state mechanism or a generic empirical PP-versus-TP optimum. Fixing stock IR duplication alone is not novelty.",
        ["reports/c14_cyclic_prefill_reopening_audit_v1.md", "reports/pipeline_alias_storage_audit_v1.md", "artifacts/pipeline_alias_cpu_v1.json", "artifacts/dag_checks/structural_matrix.json"]),
    "C15": (
        "Legal value/position-specific reassociation is real-arithmetic exact. Static one-query RoPE absorption is invalid. BF16 parity unresolved.",
        "DeepSeek MLA projection absorption/RoPE boundary, generic attention reassociation and residual restoration.",
        "Constructed RoPE test rejects static absorption. Retained rotated K plus X removes the unsharded capacity saving; head TP worsens replicated-X storage.",
        "Six CPU assertion groups and analytic geometry. No candidate model numerics or performance.",
        "No candidate execution.", "No candidate execution.", "Replicated-X saving disappears in modeled accounting.",
        "Modeled replicated X uses twice baseline per-rank KV state.",
        "A new exact position-aware contraction, BF16/gate/KV qualification, and complete same-P cost. Card's retained-K GENERIC_ONLY differs from ledger's scoped NO_NET_HEADROOM.",
        ["reports/residual_input_reassociation_redteam.md"]),
    "C16": (
        "Intended original graph/gate/KV preservation. Batch-shape BF16 parity untested.",
        "WaveFront full-depth unions, CDB, NanoFlow, generic per-row paged attention.",
        "Generic union equals the candidate union in eight constructed frontiers. Four negative controls preserve readiness rules.",
        "CPU source/frontier and address checks, not model execution.", "Legal multi-request union, no added width beyond the generic control.",
        "Actual preceding gates remain required. No candidate original-gate run.",
        "Ownership benefit NOT_RUN.", "Ownership benefit NOT_RUN.",
        "Paired online ready traces, charged cross-owner movement, full-KV union and TP/DP baselines, model numerics and quality.",
        ["reports/mixed_depth_frontier_redteam.md"]),
    "C17": (
        "Real-arithmetic tile contraction. Simple rewrite fails constructed BF16 equality.",
        "FlashNorm and generic tile/dataflow fusion.",
        "CPU BF16 cancellation rejects universal cast-boundary equivalence. Primary prior covers the current valid transformation.",
        "CPU/source only, not either checkpoint's candidate inference.", "Candidate NOT_RUN.", "Candidate NOT_RUN.",
        "Candidate NOT_RUN.", "Candidate NOT_RUN.",
        "A distinct physical mechanism, declared numerical contract, actual occupancy/join costs, and same-P net result.",
        ["reports/candidate_round3_independent_axes.md", "reports/tile_dataflow_prior_audit.md"]),
    "C18": (
        "Conditional exact reuse after complete read-set equality and side-effect publication.",
        "Generic memoization and duplicate-expression elimination.",
        "Bounded MLP screen finds zero whole-row matches: 2304 prefill and 144 decode row pairs per size. No universal no-hit theorem.",
        "CPU analysis of selected actual fixed/adaptive BF16 MLP captures from two continuity prompts per checkpoint.",
        "Selected fixed captured states only. No runtime coalescer.", "Selected adaptive captured states only. No runtime coalescer or original-gate candidate execution.",
        "Lookup/ownership candidate NOT_RUN.", "Lookup/ownership candidate NOT_RUN.",
        "Complete attention read-set equality, useful hit regime, hash/compare/storage/publication costs, and model quality.",
        ["artifacts/activation_identity_screen_v1.json", "reports/shared_weights_selected_computations_audit.md"]),
    "C19": (
        "Lossless BF16 word representation in the CPU key-only codec. No GPU compressed-attention correctness result.",
        "ZipServ lossless on-read BF16 decode, independent coding, alias-aware adaptive KV.",
        "Selected-key capacity shrinks, but shared-base/chain read demand grows under separate uncached R4 reconstruction. Not measured HBM traffic.",
        "One native TP1 controlled key capture per checkpoint. CPU coding only. Values are absent.",
        "R4 selected keys, exact byte reconstruction. No compressed candidate execution.",
        "No adaptive follow-up. Baseline terminal aliases must receive equal credit.",
        "No GPU codec or compressed attention.", "No GPU codec or compressed attention.",
        "Full K+V, legal base retention, real read/latency benefit versus independent compression, metadata/decoder costs and distributed ownership.",
        ["reports/lossless_kv_measurement_screen.md", "reports/lossless_kv_reopening_prior.md", "artifacts/lossless_kv_screen_v1.json"]),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_fields(text):
    sections = list(re.finditer(r"^## (\d+)\. ([^\n]+)\n", text, re.M))
    if sections:
        return {int(match[1]): {"heading": match[2], "text": text[match.end():sections[i + 1].start() if i + 1 < len(sections) else len(text)].strip()}
                for i, match in enumerate(sections)}
    fields = {}
    for match in re.finditer(r"^\| (\d+)\. ([^|]+) \| (.*) \|$", text, re.M):
        fields[int(match[1])] = {"heading": match[2].strip(), "text": match[3].strip()}
    return fields


def build():
    inputs = {}

    def read(path):
        full = ROOT / path
        data = full.read_bytes()
        inputs[path] = {"sha256": sha(data), "bytes": len(data)}
        return data.decode()

    ledger_path = "artifacts/candidate_status.jsonl"
    records = [json.loads(line) for line in read(ledger_path).splitlines() if line]
    # Retain the latest record by update time, then line order for a tie.
    latest = {}
    for index, row in enumerate(records):
        key = (row.get("updated_utc", ""), index)
        if row["id"] not in latest or key > latest[row["id"]][0]:
            latest[row["id"]] = (key, row)
    expected = {f"C{i:02}" for i in range(1, 20)}
    assert set(latest) == set(NOTES) == expected
    cards = {}
    for path in sorted((ROOT / "candidates").glob("C*.md")):
        match = re.match(r"C(\d+)", path.name)
        if not match:
            continue
        ident = f"C{int(match[1]):02}"
        assert ident not in cards
        rel = str(path.relative_to(ROOT))
        text = read(rel)
        fields = parse_fields(text)
        assert set(range(1, 21)).issubset(fields), (ident, sorted(fields))
        cards[ident] = (rel, text.splitlines()[0].removeprefix("# "), fields)
    assert set(cards) == expected
    rows = []
    for ident in sorted(expected):
        exact, prior, boundary, models, fixed, adaptive, p2, p4, unresolved, evidence = NOTES[ident]
        path, title, fields = cards[ident]
        status = latest[ident][1]
        for source in set(evidence + status.get("evidence", [])):
            read(source)
        rows.append({
            "id": ident, "card_path": path, "card_title": title,
            "status_ledger_record": status,
            "card_fields": {str(k): fields[k] for k in [1, 4, 9, 13, 15, 16, 17, 18, 20]},
            "reviewed_synthesis": {
                "exactness_class": exact, "nearest_prior": prior,
                "decisive_falsification_or_boundary": boundary,
                "observed_scope": {"model_evidence": models, "fixed_R4_evidence": fixed,
                                   "original_adaptive_evidence": adaptive,
                                   "P2": p2, "P4": p4,
                                   "P8": "No physical candidate result. Projected target only."},
                "unresolved_evidence": unresolved,
                "sources": evidence,
                "candidate_live_E2E": status.get("candidate_live_E2E", "MISSING"),
                "candidate_free_running_quality": status.get("candidate_free_running_quality", "MISSING"),
            }})
    supplemental = [
        "reports/c07_one_pass_fixture_contract_v1.md",
        "reports/c14_cyclic_prefill_reopening_audit_v1.md",
        "reports/full_core_storage_v4_checkpoint.md",
        "reports/storage_binding_and_lowering_audit_v1.md",
        "reports/full_core_storage_results.md",
        "artifacts/storage_same_alias_repeat_v1.json",
        "reports/native_tp_dp_cohort_analysis.md",
        "reports/state_transform_frontier_audit_v1.md",
    ]
    for source in supplemental:
        read(source)
    assert all(sha((ROOT / path).read_bytes()) == meta["sha256"] for path, meta in inputs.items())
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_class": "INTERIM_READ_ONLY_SYNTHESIS",
            "candidate_set": "Exactly C01–C19. IAX09 is a provisional dossier, not a twentieth registered card.",
            "raw_status_records": len(records), "latest_candidate_count": len(rows),
            "latest_status_selection": "Lexicographic ISO update timestamp, then last line for a tie.",
            "status_timestamp_range": sorted({row["status_ledger_record"].get("updated_utc", "MISSING") for row in rows}),
            "status_counts": dict(Counter(row["status_ledger_record"].get("status", "MISSING") for row in rows)),
            "live_E2E_counts": dict(Counter(row["reviewed_synthesis"]["candidate_live_E2E"] for row in rows)),
            "free_running_quality_counts": dict(Counter(row["reviewed_synthesis"]["candidate_free_running_quality"] for row in rows)),
            "source_fields_missing": [], "rows": rows,
            "interpretation_guards": [
                "Status counts are a snapshot, not a chronological funnel or a novelty count.",
                "Targets and observed evidence are separate.",
                "Source/CPU checks, component parity, teacher-forced inference and natural-task quality are separate.",
                "C07 eager fixture GPU testing is pending. No four-owner DP4 makespan is inferred.",
                "C14 control matches a generic physical trace, not just a graph representation.",
                "Storage interface and bounded prefill binding passes do not establish numerical parity, cached-decode addresses, HBM traffic, or causal speedup.",
                "No final campaign verdict or finalist selection. No new broad source search."],
            "storage_caveats": {
                "v1_v2": "Residency attribution remains invalid or unvalidated.",
                "v4": "All eight saved-graph/cache-root checks pass. Both sizes/P2/P4 still have numerical differences.",
                "same_alias_v5": "Fresh Base1.4B P2 same-arm repeat also differs. This does not identify the unique cause.",
                "observed_binding": "Separate Base1.4B P2 N1024 prefill, graph mode NONE. 1928 lifted operands and 11568 metadata checks pass. Not the timed v4 or cached decode processes.",
                "HBM_traffic": "NOT_MEASURED", "candidate_quality": "NOT_ESTABLISHED"},
            "inputs": inputs, "raw_inputs_unchanged_during_build": True,
            "builder_sha256": sha(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"created_utc": result["created_utc"], "candidates": len(result["rows"]),
                      "status_counts": result["status_counts"], "input_files": len(result["inputs"]),
                      "live_E2E": result["live_E2E_counts"], "quality": result["free_running_quality_counts"],
                      "raw_inputs_unchanged": result["raw_inputs_unchanged_during_build"]}))


if __name__ == "__main__":
    main()
