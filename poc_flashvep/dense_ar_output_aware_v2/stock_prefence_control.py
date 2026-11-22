"""Run a fixed control/fence/control diagnostic before policy timing."""
import argparse
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest
from .runner import save
from .c3_campaign import wait_file,launch

def prepare():
    source=ROOT/'references/source_capsules/v2_preemption_observer_fix_v5.zip'
    target=ROOT/'references/source_capsules/v2_stock_pre_forward_fence_v14.zip';assert not target.exists()
    with zipfile.ZipFile(source) as archive:files={n:archive.read(n) for n in archive.namelist()}
    name='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py';text=files[name].decode()
    anchor='    native_import_states={};native_export_prefix_states={};observer_preemption_resets=[]';assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\n    from poc_flashvep.dense_ar_output_aware_v2.stock_prefence_probe import install as install_v14_prefence\n    install_v14_prefence(cfg,role)')
    files[name]=text.encode();files['poc_flashvep/dense_ar_output_aware_v2/stock_prefence_probe.py']=(SOURCE/'stock_prefence_probe.py').read_bytes()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            if name.endswith('.py'):compile(data,name,'exec')
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    write(target.with_suffix('.json'),dict(time=time.time(),parent_SHA256=digest(source),SHA256=digest(target),
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        scope='One optional pre-forward device fence. No tensor snapshot,model correction or numerical threshold change.'))
    original=json.loads((ROOT/'configs/V2_C3Policy_StockToken_Cap2_Credit16_Strict8.json').read_text());jobs=[]
    for label,enabled in (('Control1',False),('PreFence',True),('Control2',False)):
        cfg=dict(original,name=f'V2_Stock_PreFence_{label}_N8_V14',source_capsule=target.name,
            source_capsule_SHA256=digest(target),diagnostic_D_pre_forward_fence=enabled)
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'jobs/stock_pre_forward_fence_control_v14.json',dict(time=time.time(),lanes=[jobs,[],[],[]],stop_on_failure=False,
        teacher='V2_C3Q_StockToken8_LocalOriginalTeacher8_Fix1_GeometryFixV6',jobs=jobs,
        hypotheses=['If device ordering contributes,a pre-forward fence can change the strict-state verdict while no-fence controls fail.',
            'If cold compiler or numerical variability dominates,the controls can disagree or all three can pass without a fence-specific effect.',
            'If the failure is independent of this device fence,all three can retain the mismatch.'],
        scope='Fixed control/fence/control on pair0/1,one source,originalN8,cut8,serial coordinator credit0 and exact suffix map. '
            'Do not promote a diagnostic pass to a primary proof. A device fence does not establish external transfer completion. '
            'No repeated search for a passing result. The original failed gate remains failed.'))
    print('PRE_FORWARD_FENCE_CBC_FROZEN',flush=True)

def analyze():
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_longbench400_pd.qualification_gate import compare
    plan=json.loads((ROOT/'jobs/stock_pre_forward_fence_control_v14.json').read_text());rows=[]
    for name in plan['jobs']:
        receipt=ROOT/'raw'/name/'dispatcher_result.json'
        if not receipt.exists():rows.append(dict(name=name,status='NOT_RUN'));continue
        dispatch=json.loads(receipt.read_text())
        if not dispatch['complete']:rows.append(dict(name=name,status='FAILED_EXECUTION'));continue
        coverage=audit(name);strict=compare(plan['teacher'],name)
        fences=json.loads((ROOT/'raw'/name/'diagnostic_D_pre_forward_fences_v14.json').read_text())
        if fences['enabled']:assert fences['rows']
        rows.append(dict(name=name,status='PASS' if coverage['passed'] and strict['passed'] else 'FAILED_STRICT',
            coverage=coverage,strict=strict,fence_enabled=fences['enabled'],fence_count=len(fences['rows'])))
    write(ROOT/'analysis/stock_pre_forward_fence_control_v14.json',dict(time=time.time(),rows=rows,evidence='DIAGNOSTIC',
        scope=plan['scope']+' Compare the complete fixed sequence. Do not infer a fence effect when controls also pass.'))
    print('PRE_FORWARD_FENCE_CBC_RESULTS',[(r['name'],r['status']) for r in rows],flush=True)

def resume():
    for lane in (0,1):wait_file(ROOT/'state'/f'c3_policy_qualification_v2_lane{lane}_complete_v4.json')
    launch('stock_pre_forward_fence_control_v14.json',allow_partial_budget=True);analyze()
    from .c3_policy import discovery
    launch(discovery())
    write(ROOT/'state/c3_policy_tuning_complete_v2.json',dict(time=time.time(),complete=True,
        policy_load_budget_SHA256=digest(ROOT/'configs/c3_policy_load_budget_v7.json'),
        prefence_diagnostic='stock_pre_forward_fence_control_v14.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','analyze','resume'));a=p.parse_args();globals()[a.action]()
