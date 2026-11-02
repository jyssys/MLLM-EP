"""Gate supported Looped runtime controls before bounded Serving discovery."""
import copy
from .common import *
from .configs import finish
from .evidence import status,compare,full_work,transfer
from .extension_discovery import enqueue
from .families import execution_family

PATH=ROOT/'state/extension_runtime_controls_20261102.json'


def huginn3900_initial():
    marker=ROOT/'state/extension_huginn_3900_sync_plan_20261102.json'
    if marker.exists():return
    source=json.loads((ROOT/'configs/qual_huginn_full_l3900_v3.json').read_text());new=[]
    for label,asynchronous,graph in [('sync_g4',False,4)]:
        c=copy.deepcopy(source);c.update(name=f'ext_huginn_l3900_{label}_qualification8_v2',phase='long_correctness',
            cap=32,P_slots=1,D_token_budget=128,D_kv_blocks=24576,asynchronous=asynchronous,graph_rows=graph,
            extension=True,execution_control_family='sync_l3900_g4',
            verify_transfer=True,state_probe=False,gpus=[4,5],timeout=1800,
            control_scope='The requested 3900/32 cell with supported D128 and async/graph controls. The uniform BF16 cache remains unchanged. Four full resident contexts do not fit.')
        c=finish(c,8);new.append(c)
    enqueue(new);write(marker,dict(time=time.time(),configs=[c['name'] for c in new],
        scope='Prospective supported-runtime control on the already requested maximum Huginn input/shortest output cell. No context extension or output shortening.'))


def terminal(names):
    from .campaign import terminal as observed
    return all(observed(name) in ('COMPLETE','FAILED','SKIPPED') for name in names)


def checked(reference,name,byte=False):
    result=compare(reference,name) if status(reference)==status(name)=='COMPLETE' else dict(passed=False,status=status(name),reference_status=status(reference))
    if result['passed']:
        result['work']=full_work(name);result['passed'] &= result['work']['passed']
        if byte:
            result['bytes']=transfer(name);result['passed'] &= result['bytes']['passed']
    return result


def memory_initial():
    p=ROOT/'state/extension_ouro_memory_plan_20261102.json'
    if p.exists():return
    source=json.loads((ROOT/'configs/qual_ouro_d128_kv25088_l8192_o32_v1.json').read_text());new=[]
    for budget in (32768,65536):
        for chunk in (8192,4096):
            c=copy.deepcopy(source);family=f'P8_B{budget}_K17408'
            c.update(name=f'ext_ouro_{family.lower()}_c{chunk}_qualification8_v1',phase='long_correctness',P_slots=8,
                token_budget=budget,kv_blocks=17408,chunk=chunk,chunk_label='full' if chunk==8192 else str(chunk),
                atomic_prompt=chunk==8192,gpus=[0,1],extension=True,execution_control_family=family,timeout=900,
                control_scope='Supported larger P token budget after right-sizing P KV to its eight legal source slots. D memory and native transport remain unchanged. A failed memory gate is not a performance result.')
            new.append(c)
    enqueue(new);write(p,dict(time=time.time(),configs=[c['name'] for c in new]))


def context16k32_initial():
    path=ROOT/'state/extension_ouro_16k32_plan_20261102.json'
    if path.exists():return
    source=json.loads((ROOT/'configs/qual_ouro_optional16k_d128_kv25088_v1.json').read_text());new=[]
    for label,worker,gpus,chunk in [('native','rlt_worker',[0,1],16384),('local','reference_worker',[6],16384),('chunk4096','rlt_worker',[2,3],4096)]:
        c=copy.deepcopy(source);c.update(name=f'ext_ouro_l16384_o32_{label}_qualification8_v1',worker=worker,gpus=gpus,
            cap=32,request_caps=[32]*8,phase='long_correctness',chunk=chunk,chunk_label='full' if chunk==16384 else str(chunk),
            P_slots=2,kv_blocks=17408,D_token_budget=128,D_kv_blocks=25088,reference_batch_limit=4,
            verify_transfer=worker=='rlt_worker',state_probe=True,extension=True,execution_control_family='P2_L16384_O32',timeout=1800)
        c=finish(c,8);new.append(c)
    enqueue(new);write(path,dict(time=time.time(),configs=[c['name'] for c in new],
        scope='Prospective new input/output combination in the user extension: supported 16K input and already declared 32-token output. Keep the original 16K/100 results. Reuse the frozen 16K SLO, not a new favorable threshold.'))
    write(ROOT/'configs/extension_requested_cells_20261102.json',[dict(model='ouro',length=16384,cap=32,
        declared_before_GPU=True,scope='Additional controlled long-input/short-output cell. Not the original optional 16K/100 cell.')])


def advance():
    import importlib
    from . import p_duty
    importlib.reload(p_duty);p_duty.advance()
    from .extension_effective_runtime_audit import apply
    apply()
    huginn3900_initial()
    from .analyze import capacity
    from .native_capacity_controls import analyze as native_D
    state=json.loads(PATH.read_text()) if PATH.exists() else {};changed=False
    references=json.loads((ROOT/'configs/capacity_references.json').read_text())
    candidates=[]
    for family in ('sync_g4',):
        candidates.append(('huginn',family,3500,f'ext_huginn_{family}_qualification8_v1',
            'qual_huginn_d128_kv28288_l3500_o32_n8_v1',True))
    candidates.append(('huginn','sync_l3900_g4',3900,'ext_huginn_l3900_sync_g4_qualification8_v2',
        'ext_huginn_l3900_sync_reference_qualification8_v1',True))
    for budget in (32768,65536):
        family=f'P8_B{budget}_K17408'
        candidates.append(('ouro',family,8192,f'ext_ouro_{family.lower()}_c8192_qualification8_v1',
            'qual_ouro_d128_kv25088_l8192_o32_v1',False))
    if (ROOT/'state/extension_ouro_16k32_plan_20261102.json').exists():
        candidates.append(('ouro','P2_L16384_O32',16384,'ext_ouro_l16384_o32_native_qualification8_v1',
            'ext_ouro_l16384_o32_local_qualification8_v1',False))
    for model,family,length,qualified,reference,is_huginn in candidates:
        key=model+'_'+family
        if key not in state:
            if not terminal([qualified,reference]):continue
            if length==16384 and status(qualified)==status(reference)=='COMPLETE':
                from .search import native_reference_qualification
                gate=native_reference_qualification(model,qualified,reference)
            else:gate=checked(reference,qualified,byte=not is_huginn or length==3900)
            # Huginn's N8 output gate supplements the separate valid-state
            # byte gate on the same native source/import path.
            if is_huginn and length!=3900:
                gate['prior_byte_gate']=checked('qual_huginn_d128_kv28288_l3500_memory_v1','ext_huginn_async_memory_qualification2_v1',True)
                gate['passed'] &= gate['prior_byte_gate']['passed']
            if not gate['passed']:state[key]=dict(stage='BLOCKED_GATE8',gate=gate);changed=True;continue
            cfg=json.loads((ROOT/'configs'/(qualified+'.json')).read_text());new=[];costs=[]
            c=copy.deepcopy(cfg);c.update(name=f'capacity_{model}_ext_{family.lower()}_full_p_v1',worker='capacity_worker',
                phase='independent_capacity',capacity_mode='P_ONLY',verify_transfer=False,state_probe=False,gpus=[0])
            c['jobs']=[dict(batch=b,cap=1,repeats=3) for b in ((1,2) if is_huginn else (1,2,4,8))]
            new.append(c);costs.append(c['name'])
            chunk=2048 if is_huginn else 4096
            chunk_gate=f'ext_huginn_{family}_c2048_qualification8_v1' if is_huginn else 'ext_ouro_l16384_o32_chunk4096_qualification8_v1' if length==16384 else f'ext_ouro_{family.lower()}_c4096_qualification8_v1'
            if is_huginn:
                c=copy.deepcopy(cfg);c.update(name=chunk_gate,chunk=chunk,chunk_label=str(chunk),atomic_prompt=False,verify_transfer=False,gpus=[4,5])
                new.append(c)
            if is_huginn or length==16384:
                batch=3 if is_huginn else 4
                for start in range(3):
                    c=copy.deepcopy(cfg);c.update(name=f'native_D_huginn_ext_{family}_b3_s{start}_v1',worker='prepared_d_worker',phase='independent_native_D',
                        capacity_mode='NATIVE_PREPARED_D',count=batch,P_slots=batch,prompt_ids=cfg['prompt_ids'][:batch],request_caps=[32]*batch,arrival_offsets=[0.]*batch,
                        prepare_D_only=True,prepared_D_arrival_offsets=[0.]*batch,D_ready_pattern='ALL_READY_TOGETHER',
                        verify_transfer=False,state_probe=False,independent_start=start,gpus=[2*start,2*start+1],
                        native_capacity_reference=qualified,native_capacity_control=True)
                    c['name']=f'native_D_{model}_ext_{family.lower()}_b{batch}_s{start}_v1'
                    c['data_scope']=dict(c['data_scope'],offered_count=batch)
                    new.append(c);costs.append(c['name'])
            state[key]=dict(stage='COSTS_AND_CHUNK_GATE',model=model,family=family,length=length,qualified=qualified,
                reference=reference,chunk=chunk,chunk_gate=chunk_gate,costs=costs,gate=gate,source_config=cfg)
            enqueue(new);changed=True
        item=state[key]
        if is_huginn and length==3500 and item['stage'] in ('COSTS_AND_CHUNK_GATE','COSTS_COMPLETE_PENDING') and not item.get('native_D4_controls'):
            cfg=item['source_config'];new=[]
            for start in range(3):
                c=copy.deepcopy(cfg);c.update(name=f'native_D_huginn_ext_{family}_b4_s{start}_acked_v1',worker='prepared_d_worker',
                    phase='independent_native_D',capacity_mode='NATIVE_PREPARED_D',count=4,P_slots=3,
                    prompt_ids=cfg['prompt_ids'][:4],request_caps=[32]*4,arrival_offsets=[0.]*4,prepare_D_only=True,
                    prepared_D_arrival_offsets=[0.]*4,D_ready_pattern='ALL_READY_TOGETHER',prepared_release_P_on_each_import=True,
                    verify_transfer=start==0,state_probe=False,independent_start=start,gpus=[2*start,2*start+1],
                    native_capacity_reference=item['qualified'],native_capacity_control=True,
                    capacity_scope='Native D4 with safe per-import P ACK during preparation. The D barrier still waits for all four valid imports. No D work runs before that barrier. Timed native Decode is unchanged.')
                c['data_scope']=dict(c['data_scope'],offered_count=4);new.append(c)
            enqueue(new);item['native_D4_controls']=[c['name'] for c in new];item['costs']+=item['native_D4_controls'];changed=True
        if item['stage']=='COSTS_AND_CHUNK_GATE' and terminal(item['costs']+[item['chunk_gate']]):
            gate=checked(item['qualified'],item['chunk_gate'],byte=not is_huginn)
            item['chunk_gate_result']=gate
            if gate['passed']:
                c=copy.deepcopy(item['source_config']);c.update(name=f'capacity_{model}_ext_{family.lower()}_c{item["chunk"]}_p_v1',
                    worker='capacity_worker',phase='independent_capacity',capacity_mode='P_ONLY',chunk=item['chunk'],chunk_label=str(item['chunk']),
                    atomic_prompt=False,verify_transfer=False,state_probe=False,gpus=[0])
                c['jobs']=[dict(batch=b,cap=1,repeats=3) for b in ((1,2) if is_huginn else (1,2,4,8))]
                enqueue([c]);item['costs'].append(c['name'])
            item['stage']='COSTS_COMPLETE_PENDING';changed=True
        if item['stage']=='COSTS_COMPLETE_PENDING' and terminal(item['costs']):
            samples=capacity();native=native_D();cfg=item['source_config'];family_actual=execution_family(cfg)
            p=[r for r in samples if r['model']==model and r['length']==length and r['capacity_mode']=='P_ONLY' and r['execution_family']==family_actual]
            legal_D=[r for r in native if r['model']==model and r['length']==length and r['output_cap']==32 and r['independent_starts']>=3 and
                r['D_kv_blocks']==cfg['D_kv_blocks'] and r['D_token_budget']==cfg['D_token_budget'] and
                r['asynchronous']==cfg['asynchronous'] and r['graph_rows']==cfg.get('graph_rows',cfg['capacity'])]
            d=max(legal_D,key=lambda r:r['finite_requests_per_s_mean']) if legal_D else None
            if not p or not d:item.update(stage='BLOCKED_COST_REFERENCE',P_samples=len(p),D_reference=d);changed=True;continue
            cell=copy.deepcopy(references[model+'_L'+str(length)]);chunks={}
            for chunk in sorted({r['chunk'] for r in p}):
                values=[r for r in p if r['chunk']==chunk];best=max(v['finite_requests_per_s_mean'] for v in values)
                pick=min((v for v in values if v['finite_requests_per_s_mean']>=.95*best),key=lambda v:v['batch'])
                chunks[str(chunk)]=dict(mu_P=pick['finite_requests_per_s_mean'],P_slots=pick['batch'],source=pick)
            cell['chunks']=chunks;cell['D']={'32':dict(mu_D=d['finite_requests_per_s_mean'],source=dict(batch=d['batch'],native_sources=d['source_runs']))}
            path=f'capacity_references_{model}_{family.lower()}_extension.json';write(ROOT/'configs'/path,{model+'_L'+str(length):cell})
            mu=min(max(r['mu_P'] for r in chunks.values()),d['finite_requests_per_s_mean']);new=[]
            for chunk in sorted(map(int,chunks)):
                # A finite cohort's best batch is not the native engine's
                # optimal in-flight slot count. Transfer-draining requests
                # retain slots after compute. Keep these variables separate.
                slots=list(dict.fromkeys([cfg['P_slots'],chunks[str(chunk)]['P_slots']]))
                for P_slots in slots:
                    for factor in (.75,1.,1.25):
                        c=copy.deepcopy(cfg);c.update(name=f'ext_pilot_{model}_{family.lower()}_p{P_slots}_c{chunk}_f{str(factor).replace(".","p")}_v1',
                            phase='qps_pilot',chunk=chunk,chunk_label='full' if chunk==length else str(chunk),verify_transfer=False,state_probe=False,
                            P_slots=P_slots,gpus=[0,1],reference_capacity=mu,rate_factor=factor,
                            capacity_reference_file=path,control_scope='Qualified native in-flight-slot control. The finite cohort batch and in-flight slot count are distinct. N24 is exploratory only.')
                        c=finish(c,24,rate=mu*factor);new.append(c)
            enqueue(new);item.update(stage='SERVING_PILOTS',pilots=[c['name'] for c in new],reference_capacity=mu,capacity_reference_file=path);changed=True
        if item['stage']=='SERVING_PILOTS' and terminal(item['pilots']):item['stage']='PILOTS_COMPLETE';changed=True
    if changed:write(PATH,state)
    return state


if __name__=='__main__':advance()
