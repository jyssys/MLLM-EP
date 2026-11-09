"""Extend verified paths before the final serving matrix."""
from .qualify32 import *
from .early_online import config_for


def warm_smoke(model,token=False):
    cfg=(token_cell(model,True) if token else config_for(model,'dynamic_loop'))
    cfg.update(name=f'w_{model}_{"token" if token else "zero"}_shape_smoke8',
        intervention_method='dynamic_token' if token else 'zero',
        intervention_warmup_widths=[32,128,cfg['length']//2],
        D_candidate_widths=[32,128,cfg['length']//2],
        deep_graph_budget=32,P_prefill_graph_shapes=32,
        state_probe=False,verify_transfer=False,dedicated_handoff_copy_stream=True,
        phase='EXACT_STARTUP_SHAPE_PROTOCOL_SMOKE_NOT_FINAL_PERFORMANCE')
    return cfg


def actual_state(model,token=False):
    cfg=(token_cell(model,True,32) if token else intervention(model,4096 if model=='ouro' else 2048,32,dynamic=True))
    cfg.update(name=f'n32_{model}_actual_{"token" if token else "loop"}_state32',
        timeout=1800,startup_timeout=300,dedicated_handoff_copy_stream=True,
        deep_graph_budget=32,P_prefill_graph_shapes=32,
        phase='ACTUAL_BOUNDARY_NEGOTIATION_STATE_TEST_NOT_DECODE_SAFETY_EVIDENCE')
    return cfg


def extra_cost(model,width,loops=2,quantum=None):
    cfg=primitive(model,width,loops)
    if quantum is not None:
        cfg.update(name=f'p2_{model}_tile{width}_loops{loops}_quantum{quantum}_serial8',D_loop_quantum=quantum)
    cfg.update(deep_graph_budget=32,P_prefill_graph_shapes=32)
    return cfg


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for token in (False,True):
            cfg=warm_smoke(model,token);save(cfg);jobs.append(cfg['name'])
        for token in (False,True):
            cfg=actual_state(model,token);save(cfg);jobs.append(cfg['name'])
        for width in (128,256):
            for quantum in (1,2):
                cfg=extra_cost(model,width,2,quantum);save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';write(p,json.loads(p.read_text())+jobs)


if __name__=='__main__':main()
