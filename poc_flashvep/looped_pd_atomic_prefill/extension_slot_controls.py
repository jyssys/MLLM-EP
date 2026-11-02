"""Separate native compute-slot limits from source/destination KV credit."""
import copy
from .common import *
from .configs import finish
from .evidence import status
from .extension_runtime_controls import checked,terminal
from .extension_discovery import enqueue

PATH=ROOT/'state/extension_slot_controls_20261102.json'


def advance():
    import importlib
    from . import extension_huginn_slot_controls
    importlib.reload(extension_huginn_slot_controls);extension_huginn_slot_controls.advance()
    from . import extension_ouro_long_slots
    importlib.reload(extension_ouro_long_slots);extension_ouro_long_slots.advance()
    state=json.loads(PATH.read_text()) if PATH.exists() else dict(stage='INITIAL')
    reference='qual_ouro_d128_kv25088_l8192_o32_v1'
    bounded_path=ROOT/'state/extension_bounded_draining_plan_20261102.json'
    if not bounded_path.exists() and status(reference)=='COMPLETE':
        source=json.loads((ROOT/'configs'/(reference+'.json')).read_text());new=[]
        for chunk in (8192,4096):
            c=copy.deepcopy(source);c.update(name=f'ext_ouro_p4_drain2_credit8g_c{chunk}_qualification8_v1',
                phase='long_correctness',P_slots=4,kv_blocks=22528,
                native_transport_limits=dict(max_inflight_bytes=8*1024**3,max_draining_requests=2),
                chunk=chunk,chunk_label='full' if chunk==8192 else str(chunk),atomic_prompt=chunk==8192,
                extension=True,execution_control_family='SOURCE22528_SLOT_CONTROL',verify_transfer=False,state_probe=False,
                gpus=[2,3],timeout=1200,control_scope='Keep four compute slots but bound the additional draining slots at two. Compare against default eight draining slots at the same 8-GiB wire credit. No model or transport ownership change.')
            new.append(c)
        enqueue(new);write(bounded_path,dict(time=time.time(),configs=[c['name'] for c in new]))
    if bounded_path.exists() and state['stage'] not in ('INITIAL',) and not state.get('bounded_draining_configs'):
        names=json.loads(bounded_path.read_text())['configs'];state['configs']+=names;state['bounded_draining_configs']=names
        if state['stage'] in ('SERVING_PILOTS','PILOTS_COMPLETE'):state['stage']='QUALIFICATION8'
        write(PATH,state)
    if state['stage'] in ('SERVING_PILOTS','PILOTS_COMPLETE') and any(name not in state.get('gates',{}) for name in state.get('bounded_draining_configs',[])):
        state['stage']='QUALIFICATION8';write(PATH,state)
    if state['stage']=='INITIAL':
        if status(reference)!='COMPLETE':return state
        source=json.loads((ROOT/'configs'/(reference+'.json')).read_text());new=[]
        for slots,label,credit,chunk in [(1,'default',256*1024**2,8192),(1,'credit8g',8*1024**3,8192),
            (4,'credit8g',8*1024**3,8192),(1,'credit8g',8*1024**3,4096)]:
            c=copy.deepcopy(source);c.update(name=f'ext_ouro_source22528_p{slots}_{label}_c{chunk}_qualification8_v1',
                phase='long_correctness',P_slots=slots,kv_blocks=22528,native_transport_limits=dict(max_inflight_bytes=credit),
                chunk=chunk,chunk_label='full' if chunk==8192 else str(chunk),atomic_prompt=chunk==8192,
                extension=True,execution_control_family='SOURCE22528_SLOT_CONTROL',verify_transfer=False,state_probe=False,
                gpus=[0,1] if slots==1 else [2,3],timeout=1200,
                control_scope='Existing larger source pool and qualified D128/25088 destination. Compare P1 versus P4 and default versus 8-GiB native credit. Earlier exact native state-byte gates remain separate.')
            new.append(c)
        enqueue(new);state.update(stage='QUALIFICATION8',reference=reference,configs=[c['name'] for c in new]);write(PATH,state)
    if state['stage']=='QUALIFICATION8' and terminal(state['configs']):
        references=json.loads((ROOT/'configs/capacity_references.json').read_text());cell=copy.deepcopy(references['ouro_L8192'])
        from .native_capacity_controls import analyze
        native=next((r for r in analyze() if r['model']=='ouro' and r['length']==8192 and r['output_cap']==32 and
            r['D_kv_blocks']==25088 and r['D_token_budget']==128 and r['asynchronous'] and r['graph_rows']==8 and r['independent_starts']>=3),None)
        if not native:state.update(stage='BLOCKED_NATIVE_D_REFERENCE');write(PATH,state);return state
        cell['D']['32']=dict(mu_D=native['finite_requests_per_s_mean'],source=dict(batch=native['batch'],native_sources=native['source_runs']))
        reference_file='capacity_references_ouro_source22528_slot_extension.json';write(ROOT/'configs'/reference_file,{'ouro_L8192':cell})
        mu=max(v['mu_P'] for v in cell['chunks'].values());new=[];gates={}
        for name in state['configs']:
            gate=checked(reference,name);gates[name]=gate
            if not gate['passed']:continue
            source=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            for factor in (1.,1.25,1.5):
                c=copy.deepcopy(source);c.update(name=name.replace('_qualification8_v1',f'_f{str(factor).replace(".","p")}_pilot24_v1'),
                    phase='qps_pilot',verify_transfer=False,state_probe=False,gpus=[0,1],capacity_reference_file=reference_file,
                    reference_capacity=mu,rate_factor=factor)
                c=finish(c,24,rate=mu*factor);new.append(c)
        enqueue(new);state.update(stage='SERVING_PILOTS',pilots=[c['name'] for c in new],gates=gates,
            scope='P1 is acceptable only against the strongest P4/other-C finite reference. A slot-limited slow P setting cannot establish tuning-resistant compute saturation.');write(PATH,state)
    if state['stage']=='SERVING_PILOTS' and terminal(state['pilots']):state.update(stage='PILOTS_COMPLETE');write(PATH,state)
    return state


if __name__=='__main__':advance()
