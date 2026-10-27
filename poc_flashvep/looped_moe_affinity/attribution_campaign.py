"""Queue the user's E2E-percentage addition without interrupting existing GPU jobs."""
import json
import time
import traceback
from .common import ROOT,MODELS,write
from .launch import launch

def run(config):
    write(ROOT/'attribution_campaign_state.json',dict(status='RUNNING',current=config,time=time.time()))
    result=launch(config);print(result,flush=True)
    from .aggregate_attribution import main as aggregate
    aggregate()
    return result

def main():
    while True:
        state=ROOT/'post_state.json'
        if state.exists():
            status=json.loads(state.read_text())['status']
            if status=='FAILED':raise RuntimeError('The prior GPU control campaign failed. Preserve its evidence.')
            if status=='GPU_CONTROLS_COMPLETE':break
        time.sleep(20)
    gates=[]
    for ep in [1,4,8]:
        for model in MODELS:
            common=dict(model=model,gpus=list(range(ep)),phase='attribution',batches=[1],lengths=[128],
                policies=['static'],trace_repeats=3,attribution_mode='events',timeout_s=7200)
            if ep==1 and model.startswith('ifm'):common['allocator_conf']='expandable_segments:True'
            config=dict(common,id=f'{model}_ep{ep}_e2e_share_pilot_fine_v1',detail='fine')
            run(config)
            path=ROOT/'raw'/config['id']/'attribution_status.json'
            status=json.loads(path.read_text())[0]
            if not status['low_overhead_timing_gate']:
                config=dict(common,id=f'{model}_ep{ep}_e2e_share_pilot_coarse_v1',detail='coarse')
                run(config);status=json.loads((ROOT/'raw'/config['id']/'attribution_status.json').read_text())[0]
            gates.append(dict(status,model=model,ep=ep,pilot=config['id']))
            write(ROOT/'attribution_pilot_gates.json',gates)
            if status['low_overhead_timing_gate']:
                full=dict(common,id=f'{model}_ep{ep}_e2e_share_matrix_{config["detail"]}_v1',
                    detail=config['detail'],batches=[1,2,4,8,16],lengths=[128,512,2048],
                    skip_cells=[[1,128]],trace_repeats=1,timeout_s=43200)
                run(full)
                if ep>1:
                    control=dict(common,id=f'{model}_ep{ep}_e2e_share_generic_{config["detail"]}_v1',
                        detail=config['detail'],batches=[1,16],lengths=[128,2048],
                        policies=['source_owner','replicated_counts'],trace_repeats=1,timeout_s=14400)
                    run(control)
    write(ROOT/'attribution_campaign_state.json',dict(status='COMPLETE',time=time.time(),gates=gates,
        caveat='A failed timing gate blocks clean-E2E share claims. Raw observer shares remain explicitly labeled.'))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'attribution_campaign_state.json',dict(status='FAILED',time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
