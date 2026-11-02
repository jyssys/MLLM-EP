"""Bounded, resume-safe diagnostics for the user's 11:00 KST extension."""
import argparse
from .common import *
from .configs import finish
from .evidence import status,compare,full_work,transfer

STATE=ROOT/'state/extension_discovery_20261102.json'


def queue_priority(cfg):
    if cfg.get('P_duty_diagnostic'):return 0
    if cfg.get('extension') and cfg['model'] in ('ouro','huginn'):
        if cfg['phase'] in ('independent_capacity','independent_native_D'):return 0
        if cfg['phase']=='qps_pilot':return 1
        if cfg['phase']=='final_repeat' and cfg.get('rate_index')!=1:return 4
        return 2
    if cfg.get('ar_execution_family','').startswith('BI_COMPILED_NO_GRAPH'):
        if cfg['phase']=='qps_pilot' and cfg.get('ar_execution_family')=='BI_COMPILED_NO_GRAPH_P4' and cfg.get('rate_factor') not in (.75,1.,1.25):return 5
        if cfg['phase']=='final_repeat' and cfg['arm']=='DP2':return 5
        return 5 if cfg['phase']=='final_repeat' and cfg.get('rate_index')!=1 else 3
    if cfg['model']=='huginn' and cfg['phase']=='final_repeat' and cfg.get('rate_index')==1:return 6
    return 7


def enqueue(values):
    if not values:return
    import fcntl
    names=[save_config(c) for c in values];path=ROOT/'configs/study_plan.json'
    with (ROOT/'state/study_plan.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        old=json.loads(path.read_text())['configs'];merged=list(dict.fromkeys(names+old))
        from .campaign import terminal
        quality=[];rest=[]
        for filename in merged:
            c=json.loads((ROOT/'configs'/filename).read_text())
            target=quality if terminal(c['name'])=='PENDING' and c['phase'] in ('correctness','long_correctness','chunk_correctness','final_correctness') else rest
            target.append(filename)
        rest.sort(key=lambda filename:queue_priority(json.loads((ROOT/'configs'/filename).read_text())))
        write(path,dict(configs=quality+rest))


def initial():
    if STATE.exists():return
    from .search import configured_base
    new=[]
    for label,chunk in [('full',8192),('c4096',4096)]:
        c=json.loads((ROOT/'configs/qual_ouro_d128_kv25088_l8192_o32_v1.json').read_text())
        c.update(name=f'ext_ouro_P8_{label}_qualification8_v1',phase='long_correctness',P_slots=8,chunk=chunk,
            chunk_label='full' if chunk==8192 else str(chunk),atomic_prompt=chunk==8192,gpus=[0,1],
            extension=True,control_scope='Supported P-slot tuning at the same P token budget, D pool, graph geometry, BF16, R4, and native NIXL path.')
        new.append(c)
    c=json.loads((ROOT/'configs/qual_huginn_d128_kv28288_l3500_memory_v1.json').read_text())
    c.update(name='ext_huginn_async_memory_qualification2_v1',asynchronous=True,phase='long_correctness',gpus=[4,5],
        extension=True,control_scope='Existing asynchronous runtime with fixed R32 and qualified numerical fixes. Compatibility is not assumed.')
    new.append(c)
    for invariant in (False,True):
        family='BI_EAGER' if invariant else 'DEFAULT_EAGER'
        for start,rate in enumerate((1.1432986051,2.2865972102)):
            c=configured_base('qwen8',f'ext_qwen_{family.lower()}_pd_qualification24_s{start}_v1',8000,200,
                phase='long_correctness',gpus=[2,3] if invariant else [0,1],ar_enforce_eager=True,
                ar_batch_invariant=invariant,ar_execution_family=family,extension=True,timeout=900)
            new.append(finish(c,24,rate=rate))
        c=configured_base('qwen8',f'ext_qwen_{family.lower()}_teacher_prefix_probe_v1',8000,200,
            worker='ar_tensor_probe',phase='long_correctness',gpus=[6] if invariant else [7],
            ar_enforce_eager=True,ar_batch_invariant=invariant,ar_execution_family=family,extension=True,timeout=600)
        new.append(finish(c,2))
    enqueue(new)
    write(STATE,dict(time=time.time(),stage='INITIAL_DIAGNOSTICS',configs=[c['name'] for c in new],
        scope='Looped discovery has priority. Qwen trials are correctness diagnostics, not qualified latency comparisons. Existing final runs continue after these bounded diagnostics.'))
    print('EXTENSION_INITIAL_DIAGNOSTICS',len(new))


def gates():
    state=json.loads(STATE.read_text());results={}
    for c in ('full','c4096'):
        name=f'ext_ouro_P8_{c}_qualification8_v1'
        r=compare('qual_ouro_d128_kv25088_l8192_o32_v1',name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
        if r['passed']:
            r.update(work=full_work(name),bytes=transfer(name));r['passed'] &= r['work']['passed'] and r['bytes']['passed']
        results[name]=r
    name='ext_huginn_async_memory_qualification2_v1'
    r=compare('qual_huginn_d128_kv28288_l3500_memory_v1',name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
    if r['passed']:
        r.update(work=full_work(name),bytes=transfer(name));r['passed'] &= r['work']['passed'] and r['bytes']['passed']
    results[name]=r
    for family in ('DEFAULT_EAGER','BI_EAGER'):
        a=f'ext_qwen_{family.lower()}_pd_qualification24_s0_v1';b=f'ext_qwen_{family.lower()}_pd_qualification24_s1_v1'
        results[family]=compare(a,b) if status(a)==status(b)=='COMPLETE' else dict(passed=False,reference_status=status(a),candidate_status=status(b))
    write(ROOT/'state/extension_initial_gates_20261102.json',results)
    return results


def followups():
    marker=ROOT/'state/extension_looped_followups_20261102.json'
    previous=json.loads(marker.read_text()) if marker.exists() else dict(configs=[])
    result=gates();new=[]
    if result['ext_huginn_async_memory_qualification2_v1']['passed']:
        for label,asynchronous,graph in [('async_g8',True,8),('sync_g4',False,4),('async_g4',True,4)]:
            c=json.loads((ROOT/'configs/qual_huginn_d128_kv28288_l3500_o32_n8_v1.json').read_text())
            c.update(name=f'ext_huginn_{label}_qualification8_v1',phase='long_correctness',asynchronous=asynchronous,graph_rows=graph,
                gpus=[4,5],extension=True,execution_control_family=label,
                control_scope='Supported asynchronous/graph-capacity control. Native BF16, weights, fixed R32, valid-state ownership, and numerical fixes remain unchanged.')
            new.append(c)
    cell=json.loads((ROOT/'configs/capacity_references.json').read_text())['ouro_L8192']
    from .search import serving_config
    for label,chunk in [('full',8192),('c4096',4096)]:
        if not result.get(f'ext_ouro_P8_{label}_qualification8_v1',{}).get('passed'):continue
        for factor in (1.,1.25,1.5):
            rate=max(v['mu_P'] for v in cell['chunks'].values())*factor
            c=serving_config(cell,f'ext_pilot_ouro_P8_{label}_f{str(factor).replace(".","p")}_v1',32,chunk,'qps_pilot',24,rate=rate)
            c.update(P_slots=8,D_token_budget=128,D_kv_blocks=25088,extension=True,
                control_scope='P-slot control at the same C, token budget, D memory, and transport. N24 is exploratory only.')
            new.append(c)
    new=[c for c in new if c['name'] not in previous['configs']]
    if new:
        enqueue(new);write(marker,dict(time=time.time(),configs=previous['configs']+[c['name'] for c in new]))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('initial','gates','followups'));a=p.parse_args()
    if a.action=='initial':initial()
    elif a.action=='followups':followups()
    else:print(json.dumps(gates()))


if __name__=='__main__':main()
