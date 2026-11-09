"""Keep state coverage, actual negotiation, and adaptive diversity separate."""
from .calibration import *


def reference(model):
    cfg=configuration(model,f'n32_{model}_native_full_state32',32,state=True,method='zero')
    cfg.update(timeout=1800,startup_timeout=300,deep_graph_budget=16,P_prefill_graph_shapes=16,
        phase='N32_EXACT_NATIVE_FULL_REFERENCE_NOT_TIMING',dedicated_handoff_copy_stream=True,
        D_tile_tokens=cfg['length'],D_loop_quantum=1,minimum_quantum_ms=0.,empty_decode_budget_ms=200.)
    return cfg


def mixed(model):
    cfg=intervention(model,128,count=32)
    cuts=(1,2,3,4) if model=='ouro' else (24,28,30,31,32)
    cfg.update(name=f'n32_{model}_forced_mixed_tile128_state32',timeout=1800,startup_timeout=300,
        deep_graph_budget=16,P_prefill_graph_shapes=16,D_loop_quantum=2,
        dedicated_handoff_copy_stream=True,native_request_cuts=[cuts[i%len(cuts)] for i in range(32)],
        phase='FORCED_MIXED_FRONTIER_STATE_COVERAGE_NOT_ADAPTIVE_PROOF')
    return cfg


def main():
    prepare();names=[]
    for model in ('ouro','huginn'):
        for cfg in (reference(model),mixed(model)):
            save(cfg);names.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text());write(p,queue+names)


if __name__=='__main__':main()
