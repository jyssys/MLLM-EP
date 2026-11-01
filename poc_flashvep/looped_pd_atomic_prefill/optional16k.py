"""Qualify the optional Ouro context before capacity or Serving expansion."""
from .common import *
from .configs import finish
from .evidence import status,compare,full_work

REFERENCE='qual_ouro_local_l16384_optional_v1'
CANDIDATE='qual_ouro_full_l16384_optional_v1'


def qualification_configs():
    from .search import configured_base
    extension=ROOT/'state/optional16k_data_extension.json'
    assert extension.exists(),'Fresh optional calibration source is not ready'
    new=[]
    for name,worker,gpus in [(REFERENCE,'reference_worker',[0]),(CANDIDATE,'rlt_worker',[2,3])]:
        c=configured_base('ouro',name,16384,8,phase='long_correctness',state_probe=True,
            verify_transfer=worker=='rlt_worker',worker=worker,gpus=gpus,P_slots=2,reference_batch_limit=4,timeout=1200)
        new.append(finish(c,8))
    return new


def start_capacity():
    from .search import native_reference_qualification,enqueue,configured_base
    if status(REFERENCE)!=status(CANDIDATE) or status(CANDIDATE)!='COMPLETE':
        write(ROOT/'state/optional16k_gate.json',dict(passed=False,reference_status=status(REFERENCE),candidate_status=status(CANDIDATE)));return False
    result=native_reference_qualification('ouro',CANDIDATE,REFERENCE)
    write(ROOT/'state/optional16k_gate.json',result)
    if not result['passed']:return False
    new=[]
    for mode,worker in [('P_ONLY','capacity_worker'),('PREPARED_D','capacity_worker')]:
        name='capacity_ouro_l16384_'+('p_chunkfull_optional_v1' if mode=='P_ONLY' else 'd_optional_v1')
        c=configured_base('ouro',name,16384,100,phase='independent_capacity',worker=worker,capacity_mode=mode,gpus=[0],timeout=1200)
        c=finish(c,8);c['jobs']=[dict(batch=b,cap=1 if mode=='P_ONLY' else 100,repeats=3) for b in ((1,2,4,5,8) if mode=='PREPARED_D' else (1,2,4,8))]
        new.append(c)
    for chunk in (256,1024,2048,4096):
        c=configured_base('ouro',f'qual_ouro_optional16k_chunk{chunk}_v1',16384,8,chunk=chunk,chunk_label=str(chunk),phase='chunk_correctness',gpus=[2,3],timeout=1200)
        new.append(finish(c,8))
    return enqueue(new,'OPTIONAL16K_CAPACITY')


def calibration():
    from .search import enqueue,serving_config,references
    p=ROOT/'state/chunk_gates.json';gates=json.loads(p.read_text());entries={}
    for chunk in (256,1024,2048,4096):
        name=f'qual_ouro_optional16k_chunk{chunk}_v1'
        result=compare(CANDIDATE,name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
        if result['passed']:result['work']=full_work(name);result['passed'] &= result['work']['passed']
        entries[str(chunk)]=result
    write(ROOT/'state/optional16k_chunk_gates.json',entries)
    plan=ROOT/'state/optional16k_chunk_capacity_plan.json'
    if not plan.exists():
        from .search import configured_base
        new=[]
        for chunk,gate in entries.items():
            if not gate['passed']:continue
            c=configured_base('ouro',f'capacity_ouro_l16384_p_chunk{chunk}_optional_v1',16384,100,
                chunk=int(chunk),chunk_label=chunk,worker='capacity_worker',phase='independent_capacity',capacity_mode='P_ONLY',gpus=[0],timeout=1200)
            c=finish(c,8);c['jobs']=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4,8)];new.append(c)
        write(plan,dict(configs=[save_config(c) for c in new]))
        if new:return enqueue(new,'OPTIONAL16K_CHUNK_CAPACITY')
    # Full-span local/native bytes already passed. Failed optional chunks do
    # not become valid merely because another C passed.
    cell=references().get('ouro_L16384')
    if not cell or not cell['D']:return False
    data=json.loads((ROOT/'data/calibration_ouro_L16384_extension_v1.json').read_text())
    c=serving_config(cell,'calibration_ouro_l16384_optional_v1',100,16384,'calibration',8)
    c.update(dataset='calibration_ouro_L16384_extension_v1.json',prompt_ids=[r['prompt_id'] for r in data],calibration_serial=True,
        calibration_scope='Eight disjoint optional 16K sources. The original seven-source file remains unchanged. No held-out 16K Serving result precedes this calibration.')
    return enqueue([c],'OPTIONAL16K_CALIBRATION')


def pilots():
    from .search import enqueue,serving_config
    from .analyze import percentile
    from .evidence import rows
    from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals
    name='calibration_ouro_l16384_optional_v1';values=rows(name)
    if status(name)!='COMPLETE' or len(values)!=8:return False
    path=ROOT/'configs/frozen_SLO.json';slos=json.loads(path.read_text());key='ouro_L16384'
    value=dict(TTFT_ms=2*percentile([r['ttft_ms'] for r in values],.95),TPOT_ms=2*percentile([r['tpot_ms'] for r in values],.95),
        ITL_ms=2*percentile([v for r in values for v in token_intervals(r)[0]],.99),requests=8,calibration=name,
        rule='2x independent serial calibration p95. Optional pool uses eight sources.',frozen_before_heldout_search=time.time(),not_a_hard_guarantee=True)
    if key not in slos:
        previous=digest(path);slos[key]=value;write(path,slos)
        append(ROOT/'state/SLO_extensions.jsonl',dict(time=time.time(),previous_file_SHA256=previous,new_key=key,
            existing_thresholds_unchanged=True,reason='New optional context calibrated before its held-out Serving.'))
    cell=json.loads((ROOT/'configs/capacity_references.json').read_text())[key]
    mu=min(max(c['mu_P'] for c in cell['chunks'].values()),cell['D']['100']['mu_D']);new=[]
    for chunk in dict.fromkeys([16384,cell['tuned_chunk']]):
        label='full' if chunk==16384 else str(chunk)
        for factor in (.25,.5,.75,1.,1.25,1.5,2.):
            c=serving_config(cell,f'pilot_ouro_l16384_o100_c{label}_f{str(factor).replace(".","p")}_optional_v1',100,chunk,'qps_pilot',24,rate=mu*factor)
            c.update(reference_capacity=mu,rate_factor=factor,optional_context=True);new.append(c)
    return enqueue(new,'OPTIONAL16K_QPS')
