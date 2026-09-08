"""CPU boundary test; native GPU correctness remains a separate gate."""
import json
import torch
from torch import nn
from types import SimpleNamespace
from libra_prediction_oracle import install_prediction_oracle


class Gate(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight=nn.Parameter(weight)
        self.calls=0
    def forward(self, hidden):
        self.calls+=1
        return hidden @ self.weight.T, None


class Layer(nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.mlp=nn.Module()
        self.mlp.gate=gate
    def forward(self, hidden, next_gate=None):
        return next_gate(hidden) if next_gate is not None else self.mlp.gate(hidden)


def main():
    torch.manual_seed(7328)
    layers=nn.ModuleList([Layer(Gate(torch.randn(8,4))) for _ in range(3)])
    model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    state=install_prediction_oracle(model,source_rank=2)
    full=torch.randn(12,4)
    state["capture"]=True
    for layer in layers:
        layer.mlp.gate(full)
    state["capture"]=False
    saved={k:v.clone() for k,v in state["logits"].items()}
    hidden=torch.randn(3,4)
    ordinary=layers[0](hidden,next_gate=layers[1].mlp.gate)[0]
    assert torch.equal(ordinary,hidden @ layers[1].mlp.gate.weight.T)
    state["enabled"]=True
    before=layers[1].mlp.gate.calls
    result=layers[0](hidden,next_gate=layers[1].mlp.gate)[0]
    assert torch.equal(result,saved[1][6:9])
    assert layers[1].mlp.gate.calls==before+1
    result_all=layers[0](full*2,next_gate=layers[1].mlp.gate)[0]
    assert torch.equal(result_all,saved[1])
    assert all(torch.equal(saved[k],state["logits"][k]) for k in saved)
    assert state["calls"]==2
    # Orthogonal controlled replay: fix current logical routes in BOTH variants
    # while the original predictor remains ordinary in the frozen baseline.
    state["enabled"]=False
    state["freeze_current"]=True
    current=layers[1].mlp.gate(hidden*7)[0]
    assert torch.equal(current,saved[1][6:9])
    predictor=layers[0](hidden,next_gate=layers[1].mlp.gate)[0]
    assert torch.equal(predictor,ordinary), 'Freezing actual routes must not give baseline future knowledge'
    state["enabled"]=True
    perfect=layers[0](hidden,next_gate=layers[1].mlp.gate)[0]
    assert torch.equal(perfect,current)
    print(json.dumps({"status":"PASS","scope":"CPU_BOUNDARY_NOT_NATIVE_PARITY",
        "local_slice":True,"global_shape":True,"predictor_compute_retained":True,
        "captured_future_logits_immutable":True}))


if __name__=="__main__":main()
