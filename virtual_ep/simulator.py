"""Structural and calibrated virtual-EP simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import csv
import json
import numpy as np

from .comm_model import CommunicationScenario, matrix_endpoint_bytes
from .compute_model import ComputeModel
from .event_simulator import EventPolicy, critical_path_ms
from .mapping import ExpertOwnership, SourcePartition
from .schema import TraceBundle, invocation_slices
from .traffic import build_traffic


@dataclass(frozen=True)
class InvocationPrediction:
    request_id: int
    block_id: int
    iteration_id: int
    nfe: int
    layer_id: int
    ep_size: int
    physical_rows: int
    assignment_matrix_json: str
    unique_activation_matrix_json: str
    rank_load_json: str
    outgoing_remote_bytes_json: str
    incoming_remote_bytes_json: str
    active_experts: int
    remote_assignments: int
    remote_unique_activations: int
    remote_dispatch_bytes: int
    remote_combine_bytes: int
    mean_rank_load: float
    max_rank_load: int
    max_mean_load: float
    rank_load_cv: float
    mean_token_fanout: float
    max_token_fanout: int
    dispatch_ms: float | None
    critical_expert_ms: float | None
    combine_ms: float | None
    moe_stage_ms: float | None
    compute_envelope_distance: float | None


def simulate_trace(
    trace: TraceBundle,
    ep_size: int,
    communication: CommunicationScenario | None = None,
    compute: ComputeModel | None = None,
    event_policy: EventPolicy | None = None,
) -> list[InvocationPrediction]:
    trace.validate()
    ownership = ExpertOwnership(int(trace.metadata["num_routed_experts"]), ep_size)
    partition = SourcePartition(ep_size)
    predictions: list[InvocationPrediction] = []
    hidden_size = int(trace.metadata["hidden_size"])
    for key, selected in invocation_slices(trace.rows):
        token_positions = trace.rows["source_partition_key"][selected]
        if len(token_positions) > 1 and np.any(token_positions[1:] < token_positions[:-1]):
            raise ValueError(f"source rows are not ordered in invocation {key}")
        expert_ids = trace.rows["expert_ids"][selected]
        traffic = build_traffic(
            expert_ids,
            token_positions,
            ownership,
            partition,
            hidden_size,
        )
        rank_load = traffic.rank_expert_assignments
        outgoing_bytes, incoming_bytes = matrix_endpoint_bytes(
            traffic.unique_activation_matrix, hidden_size
        )
        mean_load = float(rank_load.mean())
        max_load = int(rank_load.max(initial=0))
        cv = float(rank_load.std() / mean_load) if mean_load else 0.0
        dispatch_ms = combine_ms = expert_ms = moe_ms = distance = None
        if communication is not None:
            dispatch_ms = communication.dispatch.latency(outgoing_bytes, incoming_bytes)
            # Reverse combine carries the same token vectors with reversed
            # endpoint direction under the baseline protocol.
            combine_ms = communication.combine.latency(incoming_bytes, outgoing_bytes)
        if compute is not None:
            rank_times = []
            distances = []
            for rank in range(ep_size):
                begin = rank * ownership.experts_per_rank
                end = begin + ownership.experts_per_rank
                latency, rank_distance, _ = compute.predict(
                    traffic.expert_assignments[begin:end]
                )
                rank_times.append(latency)
                distances.append(rank_distance)
            expert_ms = max(rank_times, default=0.0)
            distance = max(distances, default=0.0)
            if communication is not None:
                moe_ms = critical_path_ms(
                    float(dispatch_ms),
                    rank_times,
                    float(combine_ms),
                    event_policy or EventPolicy(),
                )
        predictions.append(
            InvocationPrediction(
                request_id=int(key[0]),
                block_id=int(key[1]),
                iteration_id=int(key[2]),
                nfe=int(key[3]),
                layer_id=int(key[4]),
                ep_size=ep_size,
                physical_rows=int(expert_ids.shape[0]),
                assignment_matrix_json=json.dumps(traffic.assignment_matrix.tolist()),
                unique_activation_matrix_json=json.dumps(
                    traffic.unique_activation_matrix.tolist()
                ),
                rank_load_json=json.dumps(rank_load.tolist()),
                outgoing_remote_bytes_json=json.dumps(outgoing_bytes.tolist()),
                incoming_remote_bytes_json=json.dumps(incoming_bytes.tolist()),
                active_experts=int(np.count_nonzero(traffic.expert_assignments)),
                remote_assignments=traffic.remote_assignments,
                remote_unique_activations=traffic.remote_unique_activations,
                remote_dispatch_bytes=traffic.logical_dispatch_bytes,
                remote_combine_bytes=traffic.logical_combine_bytes,
                mean_rank_load=mean_load,
                max_rank_load=max_load,
                max_mean_load=float(max_load / mean_load) if mean_load else 0.0,
                rank_load_cv=cv,
                mean_token_fanout=float(traffic.token_fanout.mean()) if len(traffic.token_fanout) else 0.0,
                max_token_fanout=int(traffic.token_fanout.max(initial=0)),
                dispatch_ms=dispatch_ms,
                critical_expert_ms=expert_ms,
                combine_ms=combine_ms,
                moe_stage_ms=moe_ms,
                compute_envelope_distance=distance,
            )
        )
    return predictions


def write_predictions(
    predictions: Iterable[InvocationPrediction],
    output: Path,
    *,
    timing_verdict: str | None = None,
) -> None:
    rows = [asdict(item) for item in predictions]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no trace invocations were simulated")
    has_timing = rows[0]["moe_stage_ms"] is not None
    if has_timing and timing_verdict != "EP2-CALIBRATED":
        raise ValueError(
            "timing projections may be labelled calibrated only after held-out "
            "EP2 validation; received " + repr(timing_verdict)
        )
    with (output / "invocations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = {
        "label": (
            f"SIMULATED-EP{rows[0]['ep_size']}-EP2-CALIBRATED"
            if has_timing
            else f"SIMULATED-EP{rows[0]['ep_size']}-STRUCTURAL-ONLY"
        ),
        "invocations": len(rows),
        "requests": len({row["request_id"] for row in rows}),
        "total_remote_dispatch_bytes": sum(row["remote_dispatch_bytes"] for row in rows),
        "total_remote_combine_bytes": sum(row["remote_combine_bytes"] for row in rows),
        "mean_max_mean_load": float(np.mean([row["max_mean_load"] for row in rows])),
        "mean_rank_load_cv": float(np.mean([row["rank_load_cv"] for row in rows])),
        "payload_accounting": {
            "logical_activation_bytes": "reported",
            "protocol_packed_bytes": "not directly observable; latency calibrated from DeepEP",
            "measured_physical_traffic_bytes": "not available on this substrate",
        },
        "excluded_cost_categories": [
            "replicated-state bridge all-gather",
            "TP communication",
            "non-MoE compute",
            "production serving overhead",
        ],
        "latency_scope": "simulated routed-MoE stage only",
    }
    if has_timing:
        numeric["total_simulated_moe_stage_ms"] = sum(row["moe_stage_ms"] for row in rows)
    (output / "summary.json").write_text(json.dumps(numeric, indent=2) + "\n")
