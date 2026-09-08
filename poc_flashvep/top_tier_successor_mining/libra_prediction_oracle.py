"""Diagnostic future-knowledge baseline for the supplied Libra consumer.

Capture vanilla gate outputs, then replace ONLY a decoder's next_gate argument.
Actual current-token routing and original predictor GEMM remain unchanged.
Planner, replication, transfers, expert math and overlap streams are untouched.
This is an offline vanilla-trace oracle, not an online implementable predictor.
"""


def install_prediction_oracle(model, source_rank, source_count=4):
    import torch
    from torch import nn
    state = {"capture": False, "enabled": False, "logits": {}, "calls": 0,
             "in_predictor":False,"actual_logits":{},"freeze_current":False}
    gates = {id(layer.mlp.gate): i for i, layer in enumerate(model.model.layers)}

    def capture(layer_id):
        def hook(module, args, output):
            if state["capture"]:
                logits = output[0] if isinstance(output, tuple) else output
                state["logits"][layer_id] = logits.detach().clone()
            elif not state["in_predictor"] and (state["enabled"] or state["freeze_current"]):
                logits = output[0] if isinstance(output, tuple) else output
                if state["freeze_current"]:
                    saved=state["logits"][layer_id]
                    if saved.shape[0]!=logits.shape[0]:
                        saved=saved.chunk(source_count,dim=0)[source_rank]
                    assert saved.shape==logits.shape
                    state["actual_logits"][layer_id]=saved
                    return (saved,*output[1:]) if isinstance(output,tuple) else saved
                # Retain a reference only. Compare after the measured forward,
                # so route-fidelity checking does not add top-k kernels to it.
                state["actual_logits"][layer_id]=logits.detach()
        return hook

    class RecordedGate(nn.Module):
        def __init__(self, original, target_layer):
            super().__init__()
            self.original, self.target_layer = original, target_layer

        def forward(self, hidden):
            # Retain the original predictor compute/cost. Future knowledge alone
            # changes the supplied prediction, not the execution work budget.
            state["in_predictor"]=True
            try:
                result = self.original(hidden)
            finally:
                state["in_predictor"]=False
            if not state["enabled"]:
                return result
            saved = state["logits"][self.target_layer]
            if saved.shape[0] != hidden.shape[0]:
                assert saved.shape[0] == hidden.shape[0] * source_count
                saved = saved.chunk(source_count, dim=0)[source_rank]
            assert saved.shape == result[0].shape
            state["calls"] += 1
            return saved, result[1]

    wrappers = {}
    for i, layer in enumerate(model.model.layers):
        layer.mlp.gate.register_forward_hook(capture(i))

        def substitute(module, args, kwargs):
            gate = kwargs.get("next_gate")
            if (state["enabled"] or state["freeze_current"]) and gate is not None:
                target = gates[id(gate)]
                assert target in state["logits"], "Capture vanilla before oracle"
                if target not in wrappers:
                    wrappers[target] = RecordedGate(gate, target)
                kwargs = dict(kwargs)
                kwargs["next_gate"] = wrappers[target]
                return args, kwargs
            return None

        layer.register_forward_pre_hook(substitute, with_kwargs=True)
    return state
