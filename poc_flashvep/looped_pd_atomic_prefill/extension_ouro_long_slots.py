"""Bound reservation parking in the already-qualified Ouro 16K/32 cell."""
import copy
from .common import *
from .configs import finish
from .extension_discovery import enqueue
from .extension_runtime_controls import checked,terminal

PATH=ROOT/'state/extension_ouro_long_slot_controls_20261102.json'


def advance():
    state=json.loads(PATH.read_text()) if PATH.exists() else dict(stage='INITIAL')
    runtime=ROOT/'state/extension_runtime_controls_20261102.json'
    families=json.loads(runtime.read_text()) if runtime.exists() else {}
    source_family=families.get('ouro_P2_L16384_O32',{})
    if state['stage']=='INITIAL':
        if source_family.get('stage')!='PILOTS_COMPLETE':return state
        source=source_family['source_config'];new=[]
        for credit_gib in (8,32):
            for chunk in (16384,4096):
                c=copy.deepcopy(source);c.update(name=f'ext_ouro_l16384_p2_drain1_credit{credit_gib}g_c{chunk}_qualification8_v1',
                    phase='long_correctness',P_slots=2,
                    native_transport_limits=dict(max_inflight_bytes=credit_gib*1024**3,max_draining_requests=1),
                    chunk=chunk,chunk_label='full' if chunk==16384 else str(chunk),atomic_prompt=chunk==16384,
                    verify_transfer=True,state_probe=True,gpus=[0,1] if credit_gib==8 else [4,5],timeout=1800,
                    execution_control_family='P2_L16384_O32_BOUNDED_DRAIN',
                    control_scope='Same qualified 16K/32 cell, source/destination pools, P2 compute, full R4, BF16, and native fragment geometry. Bound extra draining slots at one. Compare existing 8-GiB credit with a supported 32-GiB limit. A larger credit is not an allocation or a new transport protocol.')
                new.append(c)
        enqueue(new);state.update(stage='QUALIFICATION8',reference=source_family['qualified'],configs=[c['name'] for c in new]);write(PATH,state)
    if state['stage']=='QUALIFICATION8' and terminal(state['configs']):
        reference_file=source_family['capacity_reference_file']
        cell=json.loads((ROOT/'configs'/reference_file).read_text())['ouro_L16384']
        mu=max(v['mu_P'] for v in cell['chunks'].values());new=[];gates={}
        for name in state['configs']:
            gate=checked(state['reference'],name,byte=True);gates[name]=gate
            if not gate['passed']:continue
            source=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            for factor in (1.,1.25,1.5):
                c=copy.deepcopy(source);c.update(name=name.replace('_qualification8_v1',f'_f{str(factor).replace(".","p")}_pilot24_v1'),
                    phase='qps_pilot',verify_transfer=False,state_probe=False,gpus=[0,1],capacity_reference_file=reference_file,
                    reference_capacity=mu,rate_factor=factor)
                c=finish(c,24,rate=mu*factor);new.append(c)
        enqueue(new);state.update(stage='SERVING_PILOTS',pilots=[c['name'] for c in new],gates=gates,
            scope='Prospective control. Retain the original 16K pilots, strongest P service, frozen SLOs, and strict kernel/credit gates.');write(PATH,state)
    if state['stage']=='SERVING_PILOTS' and terminal(state['pilots']):state.update(stage='PILOTS_COMPLETE');write(PATH,state)
    return state


if __name__=='__main__':advance()
