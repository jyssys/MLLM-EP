"""Resume the remaining gates after dispatcher loss. Never repeat completed jobs."""
import time
from .prepare import *
from .run import execute
from .qualification import compare,compare_outputs
from .generic_heavy_followup import safe


def dispatch(cfg):
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():cfg=json.loads(path.read_text())
    else:save(cfg)
    write(ROOT/'state/recovery_controller.json',dict(time=time.time(),name=cfg['name'],stage='DISPATCH'))
    try:return execute(cfg['name'])
    except AssertionError as error:
        if not any(s in str(error) for s in ('GPU closeout boundary','Job cannot safely fit GPU budget')):raise
        write(ROOT/'state'/f'recovery_skip_{cfg["name"]}.json',dict(time=time.time(),name=cfg['name'],
            status='NOT_RUN_TIME_BUDGET',reason=str(error)))
        return False


def existing(name):
    return dispatch(json.loads((ROOT/'configs'/(name+'.json')).read_text()))


def opportunity(model,chunk,alignment=False,d_device=False):
    suffix='_d_device_v1' if d_device else '_alignment_v1' if alignment else '_v1'
    cfg=configuration(model)
    cfg.update(name=f'opportunity_{model}_c{chunk}'+suffix,worker='operator_micro',micro_opportunity=True,
        opportunity_chunk=chunk,opportunity_token_budget=512,opportunity_graph_rows=512,opportunity_strict_join=True)
    if alignment or d_device:cfg['opportunity_same_align']=True
    if d_device:cfg['gpus']=list(reversed(cfg['gpus']))
    return dispatch(cfg)


def main():
    initialize()
    name='qual32_huginn_paced_c128_v4'
    if existing(name):compare('qual32_huginn_native_v3',name,32)
    for method in ('full','chunked','paced'):
        gate=ROOT/'raw'/(f'qual32_huginn_full_scoped_v3' if method=='full' else f'qual32_huginn_{method}_c128_v4')/'local_qualification.json'
        if not gate.exists() or not json.loads(gate.read_text())['passed']:continue
        cfg=configuration('huginn',method,32 if method=='full' else 128)
        cfg.update(name=f'serve_huginn_{method}_c128_fixed25_v5',dlocal_chunk=128,dlocal_every=4,
            dlocal_max_active=1,deep_complete_D_geometry_warmup=True)
        if dispatch(cfg):compare_outputs('local_huginn_native_n128_s2026110701_off',cfg['name'],cfg['count'])
    existing('micro_huginn_strict_grid_v5')
    for model in ('ouro','huginn'):
        cfg=configuration(model)
        cfg.update(name=f'micro_{model}_kernel_profile_v5',worker='operator_micro',micro_strict_geometry=True,
            micro_cases=1,micro_grid=[dict(chunk=32,batch=2,aligned=False)],micro_profile=True)
        dispatch(cfg)
    cfg=configuration('huginn')
    cfg.update(name='micro_huginn_matched_c248_v1',worker='operator_micro',micro_strict_geometry=True,
        micro_cases=2,micro_grid=[dict(chunk=248,batch=2,aligned=a) for a in (True,False)])
    dispatch(cfg)
    for model in ('ouro','huginn'):
        for chunk in (128,248):
            gate=ROOT/'raw'/(f'micro_{model}_strict_grid_v5' if chunk==128 else f'micro_{model}_matched_c248_v1')/'micro_summary.json'
            if not gate.exists() or not json.loads(gate.read_text())['all_micro_states_exact']:continue
            opportunity(model,chunk)
            opportunity(model,chunk,alignment=True)
        gate=ROOT/'raw'/f'micro_{model}_matched_c248_v1/micro_summary.json'
        if gate.exists() and json.loads(gate.read_text())['all_micro_states_exact']:
            opportunity(model,248,d_device=True)
    for model in ('ouro','huginn'):
        prior=f'serve_{model}_paced_c128_fixed25_v5'
        if safe(prior,model) is not True:continue
        cfg=configuration(model,'paced',rate=6.356 if model=='ouro' else 1.971)
        cfg.update(name=f'serve_{model}_paced_heavy_fixed25_'+('clean_restart_v2' if model=='ouro' else 'v1'),
            dlocal_chunk=128,dlocal_every=4,dlocal_max_active=1,
            deep_complete_D_geometry_warmup=True,load='heavy',source_medium_control=prior)
        dispatch(cfg)
    write(ROOT/'state/recovery_controller.json',dict(time=time.time(),stage='COMPLETE'))


if __name__=='__main__':main()
