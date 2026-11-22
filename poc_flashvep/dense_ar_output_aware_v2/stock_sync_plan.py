"""Freeze a clean repeat and a fence-only counterfactual for the exact N8 case."""
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest
from .runner import save

def prepare():
    parent=ROOT/'references/source_capsules/v2_preemption_observer_fix_v5.zip'
    target=ROOT/'references/source_capsules/v2_stock_fence_diagnostic_v11.zip';assert not target.exists()
    with zipfile.ZipFile(parent) as archive:files={n:archive.read(n) for n in archive.namelist()}
    name='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py';text=files[name].decode()
    anchor='    native_import_states={};native_export_prefix_states={};observer_preemption_resets=[]';assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\n    from poc_flashvep.dense_ar_output_aware_v2.stock_sync_probe import install as install_stock_sync_probe\n    install_stock_sync_probe(cfg,role)')
    files[name]=text.encode();files['poc_flashvep/dense_ar_output_aware_v2/stock_sync_probe.py']=(SOURCE/'stock_sync_probe.py').read_bytes()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            if name.endswith('.py'):compile(data,name,'exec')
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    write(target.with_suffix('.json'),dict(parent_SHA256=digest(parent),SHA256=digest(target),
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Optional post-forward device fence only. No weights,model math,inputs,outputs or sampling are corrected. Diagnostic use.'))
    prior='V2_C3Q_StockToken2_Strict8_CoordinatorFix1_GeometryFixV6'
    original=json.loads((ROOT/'configs'/(prior+'.json')).read_text());assert original['gpus']==[2,3]
    jobs=[]
    for label in ('NoProbeRepeat','FenceOnly'):
        cfg=dict(original);name=f'V2_StockSyncControl_Token2_N8_{label}_V11';cfg['name']=name
        if label=='FenceOnly':cfg.update(source_capsule=target.name,source_capsule_SHA256=digest(target),
            stock_sync_after_forward_roles=['prefill','decode'])
        save(cfg);jobs.append(name)
    write(ROOT/'jobs/stock_sync_controls_v11.json',dict(time=time.time(),lanes=[[],jobs,[],[]],stop_on_failure=False,
        teacher='V2_C3Q_StockToken2_LocalOriginalTeacher8_Fix1_GeometryFixV6',
        scope='Repeat the actual failing N8 case without extra observation. Then add only a post-forward completion fence. '
            'Keep the same pair2/3,original requests,arrival,cut,suffix map,compiler,source5 baseline and output contract. '
            'The source11 extension does not change an arithmetic operator. '
            'These tests discriminate timing/lifetime effects from numerical tensor-copy observations. '
            'A pass is not a primary performance qualification or proof of one specific race.'))
    print('STOCK_FENCE_ONLY_COUNTERFACTUAL_FROZEN',digest(target),flush=True)

if __name__=='__main__':prepare()
