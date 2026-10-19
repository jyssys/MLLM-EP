"""Opt-in argument observations in the retained-graph debug worker only."""
import hashlib
import json
import os
from pathlib import Path

from poc_ouro_parallelism_deep_discovery.cuda_graph_argument_getter import (
    OPT_IN, ReadOnlyArgumentRuntime, validate_policy)

if os.environ.get(OPT_IN) != "1":
    raise RuntimeError(f"Set {OPT_IN}=1 before importing this diagnostic worker.")

from poc_ouro_parallelism_deep_discovery import graph_debug_probe as debug
from poc_ouro_parallelism_deep_discovery.graph_typed_edge_probe import TypedEdgeGraphDebugProbe
from poc_ouro_parallelism_deep_discovery.operand_binding import parameter_snapshot


class ArgumentGraphDebugProbe(TypedEdgeGraphDebugProbe):
    def graph_argument_export(self, layers, policy):
        """Inspect templates after one completed request. Do not modify or replay graphs."""
        policy = validate_policy(policy)
        # The existing export ends observation, synchronizes, and retains replay identities.
        result = super().graph_debug_export(layers)
        runtime = ReadOnlyArgumentRuntime()
        model = self.get_model()
        named = list(model.named_parameters(remove_duplicate=False))
        parameters = parameter_snapshot(named)
        for name, value in named:
            parameters[name]["device"] = str(value.device)
        parameter_digest = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        for record in result["records"]:
            graph = debug._graphs[record["graph_object_id"]]
            raw, executable = int(graph.raw_cuda_graph()), int(graph.raw_cuda_graph_exec())
            payload = runtime.observe(raw, policy, parameters)
            if int(graph.raw_cuda_graph()) != raw or int(graph.raw_cuda_graph_exec()) != executable:
                raise RuntimeError("A retained graph handle changed during argument export.")
            payload.update(graph_object_id=record["graph_object_id"], observed_replays=record["replay_count"],
                rank=result["rank"], observed_executable_handle=hex(executable),
                executable_handle_scope="Identity only. No executable parameters were queried.",
                parameter_snapshot=parameters, parameter_snapshot_sha256=parameter_digest,
                serialization_scope="Worker RPC after completed request and device synchronization. No graph mutation in this RPC.",
                executable_update_audit="NOT_PERFORMED. Stable handles and topology do not prove unchanged executable arguments.",
                source_dot=record["path"])
            dot = Path(record["path"])
            path = dot.with_name(dot.stem + "_arguments.json")
            with path.open("x") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            record["argument_observations"] = dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), coverage=payload["coverage"],
                executed_operand_identity="UNRESOLVED", timing="NOT_MEASURED")
        result.update(argument_metadata=True, argument_runtime_library=runtime.path,
            argument_runtime_version=runtime.version, argument_policy=policy,
            scope="Retained source-template arguments for actual replay identities. No proof of executed reads or timing.")
        return result
