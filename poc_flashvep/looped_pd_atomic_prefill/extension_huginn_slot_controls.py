"""Test supported bounded source draining without changing Huginn compute."""
import copy
from .common import *
from .configs import finish
from .evidence import status
from .extension_runtime_controls import checked,terminal
from .extension_discovery import enqueue

PATH=ROOT/'state/extension_huginn_slot_controls_20261102.json'


def advance():
    state=json.loads(PATH.read_text()) if PATH.exists() else dict(stage='INITIAL')
    reference='ext_huginn_sync_g4_qualification8_v1'
    if state['stage']=='INITIAL':
        if status(reference)!='COMPLETE':return state
        source=json.loads((ROOT/'configs'/(reference+'.json')).read_text());new=[]
        for draining in (0,1):
            for chunk in (3500,2048):
                c=copy.deepcopy(source);c.update(name=f'ext_huginn_p1_drain{draining}_credit8g_c{chunk}_qualification8_sync_v2',
                    phase='long_correctness',P_slots=1,native_transport_limits=dict(max_inflight_bytes=8*1024**3,max_draining_requests=draining),
                    chunk=chunk,chunk_label='full' if chunk==3500 else str(chunk),atomic_prompt=chunk==3500,
                    verify_transfer=False,state_probe=False,gpus=[0,1] if draining==0 else [4,5],timeout=1200,
                    execution_control_family='HUGINN_SOURCE_DRAIN_CONTROL_SYNC',
                    control_scope='Same qualified Huginn Sync/graph-4 computation, original P pool, D128/28288 pool, and native 8-GiB credit. Compare zero or one extra source draining slot against the unchanged default-eight control. No shortened outputs or altered KV.')
                new.append(c)
        enqueue(new);state.update(stage='QUALIFICATION8',reference=reference,configs=[c['name'] for c in new]);write(PATH,state)
    if state['stage']=='QUALIFICATION8' and terminal(state['configs']):
        runtime_path=ROOT/'state/extension_runtime_controls_20261102.json'
        runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {};cost=runtime.get('huginn_sync_g4',{})
        if not cost.get('capacity_reference_file'):return state
        from .native_capacity_controls import analyze
        native=[v for v in analyze() if v['model']=='huginn' and v['length']==3500 and v['batch']==4 and v['independent_starts']>=3 and
            v['D_kv_blocks']==28288 and v['D_token_budget']==128 and not v['asynchronous'] and v['graph_rows']==4]
        if not native:
            if terminal(cost.get('native_D4_controls',[])):
                state.update(stage='BLOCKED_NATIVE_D4_REFERENCE');write(PATH,state)
            return state
        d=max(native,key=lambda r:r['finite_requests_per_s_mean']);references=json.loads((ROOT/'configs'/cost['capacity_reference_file']).read_text())
        cell=references['huginn_L3500'];muP=max(v['mu_P'] for v in cell['chunks'].values())
        if d['finite_requests_per_s_mean']<1.2*.85*muP:
            state.update(stage='NO_STRICT_RESERVE_AT_NEAR_P_SERVICE',muP=muP,muD=d['finite_requests_per_s_mean'],
                scope='Even the all-ready native D4 rate cannot clear the existing 20% reserve at 85% of strongest P service. Do not slow Prefill to force a positive label.');write(PATH,state);return state
        cell['D']['32']=dict(mu_D=d['finite_requests_per_s_mean'],source=dict(batch=4,native_sources=d['source_runs']))
        reference_file='capacity_references_huginn_bounded_drain_extension.json';write(ROOT/'configs'/reference_file,references)
        mu=min(muP,d['finite_requests_per_s_mean']);new=[];gates={}
        for name in state['configs']:
            gate=checked(reference,name);gates[name]=gate
            if not gate['passed']:continue
            source=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            for factor in (.85,1.,1.15):
                c=copy.deepcopy(source);c.update(name=name.replace('_qualification8_sync_v2',f'_f{str(factor).replace(".","p")}_pilot24_sync_v2'),
                    phase='qps_pilot',gpus=[0,1],capacity_reference_file=reference_file,reference_capacity=mu,rate_factor=factor)
                c=finish(c,24,rate=mu*factor);new.append(c)
        enqueue(new);state.update(stage='SERVING_PILOTS',pilots=[c['name'] for c in new],gates=gates,muP=muP,
            muD=d['finite_requests_per_s_mean']);write(PATH,state)
    if state['stage']=='SERVING_PILOTS' and terminal(state['pilots']):state.update(stage='PILOTS_COMPLETE');write(PATH,state)
    return state


if __name__=='__main__':advance()
