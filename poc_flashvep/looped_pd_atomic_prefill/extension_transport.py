"""Repeat supported native wire-credit tuning after workspace qualification."""
import copy
from .common import *
from .configs import finish
from .evidence import status
from .extension_discovery import enqueue
from .extension_runtime_controls import checked,terminal

PATH=ROOT/'state/extension_transport_20261102.json'


def advance():
    import importlib
    from . import extension_slot_controls
    importlib.reload(extension_slot_controls);extension_slot_controls.advance()
    state=json.loads(PATH.read_text()) if PATH.exists() else {};changed=False
    sources={'ouro':('ext_ouro_p8_b32768_k17408_c8192_qualification8_v1','P8_B32768_K17408'),
        'huginn':('ext_huginn_sync_g4_qualification8_v1','sync_g4')}
    controls_path=ROOT/'state/extension_runtime_controls_20261102.json'
    controls=json.loads(controls_path.read_text()) if controls_path.exists() else {}
    for model,(reference,family) in sources.items():
        if model not in state:
            if status(reference)!='COMPLETE':continue
            source=json.loads((ROOT/'configs'/(reference+'.json')).read_text());new=[]
            for label,limit in [('credit1g',1024**3),('credit8g',8*1024**3)]:
                version='sync_v2' if model=='huginn' else 'v1'
                c=copy.deepcopy(source);c.update(name=f'ext_wire_{model}_{label}_qualification8_{version}',
                    phase='long_correctness',verify_transfer=True,state_probe=model!='huginn',gpus=[0,1] if model=='ouro' else [4,5],
                    native_transport_limits=dict(max_inflight_bytes=limit),timeout=1800,
                    control_scope='Existing PDConfig credit only. Same 64-MiB fragmentation, qualified graph/async state, source pool, destination pool, and BF16. Not a new migration method.')
                new.append(c)
            enqueue(new);state[model]=dict(stage='QUALIFICATION8',reference=reference,family=family,
                source=source,qualification=[c['name'] for c in new]);changed=True
        item=state[model]
        control=controls.get(model+'_'+family)
        if item['stage']=='QUALIFICATION8' and terminal(item['qualification']) and control and control.get('capacity_reference_file'):
            reference_file=control['capacity_reference_file'];cell=json.loads((ROOT/'configs'/reference_file).read_text())[model+'_L'+str(item['source']['length'])]
            mu=min(max(v['mu_P'] for v in cell['chunks'].values()),cell['D']['32']['mu_D']);new=[];gates={}
            for name in item['qualification']:
                gate=checked(reference,name,byte=True);gates[name]=gate
                if not gate['passed']:continue
                source=json.loads((ROOT/'configs'/(name+'.json')).read_text());label='credit1g' if source['native_transport_limits']['max_inflight_bytes']==1024**3 else 'credit8g'
                for chunk in sorted(map(int,cell['chunks'])):
                    for factor in (1.,1.25):
                        version='sync_v2' if model=='huginn' else 'v1'
                        c=copy.deepcopy(source);c.update(name=f'ext_wire_{model}_{label}_c{chunk}_f{str(factor).replace(".","p")}_pilot24_{version}',
                            phase='qps_pilot',chunk=chunk,chunk_label='full' if chunk==source['length'] else str(chunk),
                            verify_transfer=False,state_probe=False,gpus=[0,1],reference_capacity=mu,rate_factor=factor,
                            capacity_reference_file=reference_file)
                        c=finish(c,24,rate=mu*factor);new.append(c)
            enqueue(new);item.update(stage='SERVING_PILOTS',pilots=[c['name'] for c in new],gates=gates);changed=True
        if item['stage']=='SERVING_PILOTS' and terminal(item['pilots']):item['stage']='PILOTS_COMPLETE';changed=True
    if changed:write(PATH,state)
    return state


if __name__=='__main__':advance()
