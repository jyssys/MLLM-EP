"""Queue limited geometry coverage and independent cost calibration."""
from .followups import *


def coverage(model):
    cfg=intervention(model,128,state=True)
    cfg.update(name=f'n8_{model}_variable_geometry_state8',
        native_request_cuts=([1,2,3,4]*2 if model=='ouro' else [24,28,30,31,32,24,28,30]),
        qualification_tile_widths=[16,32,64,128,256,512,1024,cfg['length']],
        D_loop_quantum=2,deep_graph_budget=32,P_prefill_graph_shapes=32,
        dedicated_handoff_copy_stream=True,
        phase='FORCED_GEOMETRY_STATE_COVERAGE_NOT_ADAPTIVE_OR_SAFETY_EVIDENCE')
    cfg.pop('native_partial_cut',None)
    return cfg


def token_cost(model,width,quantum):
    cfg=primitive(model,width,1)
    cfg.update(name=f'pt_{model}_suffix512_tile{width}_quantum{quantum}_serial8',
        intervention_method='static_token',native_frontier=[0,cfg['length']-512],
        D_loop_quantum=quantum,deep_graph_budget=32,P_prefill_graph_shapes=32,
        dedicated_handoff_copy_stream=True)
    cfg.pop('native_partial_cut',None)
    return cfg


def deferred_token(model):
    cfg=token_cell(model,True)
    cfg.update(name=f'n8_{model}_deferred_token_state8',
        defer_token_arming=True,initial_token_budget_selection=True,
        empty_decode_budget_ms=1000.,dedicated_handoff_copy_stream=True,
        deep_graph_budget=32,P_prefill_graph_shapes=32)
    return cfg


def main():
    prepare();jobs=[]
    # A syntax error in a frozen source snapshot is not a GPU correctness result.
    cfg=actual_state('ouro',True)
    cfg['name']='n32_ouro_actual_token_state32_v2'
    save(cfg);jobs.append(cfg['name'])
    for model in ('ouro','huginn'):
        for cfg in (coverage(model),deferred_token(model)):
            save(cfg);jobs.append(cfg['name'])
        for width,quantum in ((32,1),(128,2),(256,2)):
            cfg=token_cost(model,width,quantum);save(cfg);jobs.append(cfg['name'])
        for loops in ((2,3) if model=='ouro' else (2,4,8)):
            cfg=primitive(model,4096 if model=='ouro' else 2048,loops)
            cfg.update(name=f'pf_{model}_loops{loops}_quantum1_serial8',
                D_loop_quantum=1,deep_graph_budget=32,P_prefill_graph_shapes=32,
                dedicated_handoff_copy_stream=True)
            save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';write(p,json.loads(p.read_text())+jobs)


if __name__=='__main__':main()
