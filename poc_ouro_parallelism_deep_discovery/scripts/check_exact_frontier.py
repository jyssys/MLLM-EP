"""Check bounded operator dependencies. This is not a latency simulator."""

import json


class Graph:
    def __init__(self):
        self.pred = {}
        self.owner = {}

    def add(self, name, predecessors=(), owner="external"):
        assert name not in self.pred
        assert all(p in self.pred for p in predecessors), (name, predecessors)
        self.pred[name] = set(predecessors)
        self.owner[name] = owner
        return name

    def before(self, source, target):
        pending = list(self.pred[target])
        seen = set()
        while pending:
            node = pending.pop()
            if node == source:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(self.pred[node])
        return False

    def independent(self, a, b):
        return not self.before(a, b) and not self.before(b, a)


def make_case(layers, requests, executed_passes, adaptive):
    graph = Graph()
    saved = {}
    probes = []
    exit_trace = []
    for q in range(requests):
        prior_sample = None
        for token in range(2):
            passes = executed_passes if not adaptive or requests == 1 else 2 + (q + token) % (executed_passes - 1)
            exit_trace.append({"request": q, "token": token, "passes": passes})
            prefix = f"q{q}/t{token}"
            compute_owner = f"compute_owner_{q % 2}"
            kv_owner = f"kv_owner_{q % 2}"
            source = graph.add(prefix + "/embed", () if prior_sample is None else (prior_sample,), compute_owner)
            previous_cdf = None
            previous_guard = None
            layer_writes = {}
            boundaries = []
            cdfs = []
            for r in range(passes):
                for layer in range(layers):
                    stem = f"{prefix}/r{r}/l{layer}/"
                    guards = []
                    if layer == 0 and r > 0 and adaptive and r >= 2:
                        guards.append(previous_guard)
                    n1 = graph.add(stem + "N1", (source, *guards), compute_owner)
                    q_node = graph.add(stem + "Q", (n1,), compute_owner)
                    k_node = graph.add(stem + "K", (n1,), compute_owner)
                    v_node = graph.add(stem + "V", (n1,), compute_owner)
                    rq = graph.add(stem + "RoPE_Q", (q_node,), compute_owner)
                    rk = graph.add(stem + "RoPE_K", (k_node,), compute_owner)
                    write = graph.add(stem + "KV_WRITE", (rk, v_node), kv_owner)
                    layer_writes[r, layer] = write
                    past = () if token == 0 else (saved[q, token - 1, r, layer],)
                    attention = graph.add(stem + "ATTENTION", (rq, write, *past), kv_owner)
                    output = graph.add(stem + "O_LOCAL", (attention,), compute_owner)
                    output_reduce = graph.add(stem + "O_REDUCE", (output,), f"tp_group_{q % 2}")
                    n2 = graph.add(stem + "N2", (output_reduce,), compute_owner)
                    y = graph.add(stem + "ATT_RESIDUAL", (source, n2), compute_owner)
                    n3 = graph.add(stem + "N3", (y,), compute_owner)
                    gate = graph.add(stem + "MLP_GATE", (n3,), compute_owner)
                    up = graph.add(stem + "MLP_UP", (n3,), compute_owner)
                    silu = graph.add(stem + "SILU", (gate,), compute_owner)
                    product = graph.add(stem + "MLP_PRODUCT", (silu, up), compute_owner)
                    down = graph.add(stem + "DOWN_LOCAL", (product,), compute_owner)
                    down_reduce = graph.add(stem + "DOWN_REDUCE", (down,), f"tp_group_{q % 2}")
                    n4 = graph.add(stem + "N4", (down_reduce,), compute_owner)
                    source = graph.add(stem + "MLP_RESIDUAL", (y, n4), compute_owner)
                    if layer == 0:
                        probes.append((q_node, k_node, gate, up, n2, output_reduce, n4, down_reduce))
                boundary = graph.add(f"{prefix}/r{r}/LOOP_NORM", (source,), compute_owner)
                boundaries.append(boundary)
                source = boundary
                if adaptive and r < 3:
                    gate_logit = graph.add(f"{prefix}/r{r}/EXIT_GATE", (boundary,), compute_owner)
                    cdf = graph.add(f"{prefix}/r{r}/CDF", (gate_logit,) if previous_cdf is None else (gate_logit, previous_cdf), "host_policy")
                    previous_cdf = previous_guard = cdf
                    cdfs.append(cdf)
            decision = graph.add(prefix + "/EXIT_DECISION", (source, previous_cdf) if adaptive and passes < 4 else (source,), "host_policy")
            for r in range(4):
                for layer in range(layers):
                    if r < passes:
                        saved[q, token, r, layer] = layer_writes[r, layer]
                    else:
                        saved[q, token, r, layer] = graph.add(f"{prefix}/r{r}/l{layer}/TERMINAL_COPY", (layer_writes[passes - 1, layer], decision), kv_owner)
            head = graph.add(prefix + "/LM_HEAD", (source, decision), compute_owner)
            prior_sample = graph.add(prefix + "/SAMPLE", (head,), compute_owner)
            for r in range(1, passes):
                next_n1 = f"{prefix}/r{r}/l0/N1"
                assert graph.before(boundaries[r - 1], next_n1)
                if adaptive:
                    if r == 1:
                        assert not graph.before(cdfs[0], next_n1)
                    else:
                        assert graph.before(cdfs[r - 1], next_n1)
            if token == 1:
                assert graph.before(f"q{q}/t0/SAMPLE", prefix + "/embed")
    for q_node, k_node, gate, up, n2, output_reduce, n4, down_reduce in probes:
        assert graph.independent(q_node, k_node)
        assert graph.independent(gate, up)
        assert graph.before(output_reduce, n2)
        assert graph.before(down_reduce, n4)
    if requests > 1:
        assert graph.independent("q0/t0/r0/l0/N1", "q1/t0/r0/l0/N1")
        assert graph.owner["q0/t0/r0/l0/N1"] != graph.owner["q1/t0/r0/l0/N1"]
    return {
        "layers": layers,
        "requests": requests,
        "tokens_per_request": 2,
        "mode": "adaptive" if adaptive else "fixed",
        "executed_passes": executed_passes,
        "exit_trace": exit_trace,
        "nodes": len(graph.pred),
        "edges": sum(map(len, graph.pred.values())),
        "status": "PASS",
    }


def main():
    cases = []
    for layers in (24, 48):
        for requests in (1, 3):
            cases.append(make_case(layers, requests, 4, False))
            for passes in (2, 3, 4):
                cases.append(make_case(layers, requests, passes, True))
    print(json.dumps({
        "test": "CPU_SYMBOLIC_EXACT_FRONTIER",
        "cases": cases,
        "case_count": len(cases),
        "checks": ["acyclic_predecessor_construction", "loop_norm_before_next_pass", "minR2_R1_gate_not_needed_before_R2", "eligible_exit_guard_before_nonprovisional_next_pass", "token_sampling_before_next_embedding", "Q_K_independence", "MLP_gate_up_independence", "TP_reduction_before_postprojection_norm", "correct_previous_token_KV_provider", "no_crossrequest_data_barrier"],
        "not_tested": ["hardware_overlap", "latency", "BF16_numerics", "gate_values", "concurrent_allocator", "speculative_commit", "live_outputs"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
