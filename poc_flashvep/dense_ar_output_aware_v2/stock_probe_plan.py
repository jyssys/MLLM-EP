"""Freeze a one-request attention-path test without changing the serving baseline."""
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest
from .runner import save
from .work_contract import regenerate

def main():
    folder=ROOT/'references/source_capsules';old=folder/'v2_preemption_observer_fix_v5.zip';new=folder/'v2_stock_attention_diagnostic_v8.zip';assert not new.exists()
    with zipfile.ZipFile(old) as archive:files={n:archive.read(n) for n in archive.namelist()}
    name='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py';text=files[name].decode()
    anchor='    native_import_states={};native_export_prefix_states={};observer_preemption_resets=[]';assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\n    from poc_flashvep.dense_ar_output_aware_v2.stock_attention_probe import install as install_stock_attention_probe\n    install_stock_attention_probe(cfg,role,info)')
    files[name]=text.encode();files['poc_flashvep/dense_ar_output_aware_v2/stock_attention_probe.py']=(SOURCE/'stock_attention_probe.py').read_bytes()
    with zipfile.ZipFile(new,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path,data in sorted(files.items()):
            if path.endswith('.py'):compile(data,path,'exec')
            entry=zipfile.ZipInfo(path,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    write(new.with_suffix('.json'),dict(parent_SHA256=digest(old),SHA256=digest(new),files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Read-only layer0 FA3 snapshots on q0. No model math or compiler setting change. Diagnostic timing only.'))
    prior=json.loads((ROOT/'configs/c3_stock_geometry_repair_v6.json').read_text());item=next(x for x in prior['items'] if x['label']=='StockToken2')
    data=json.loads((ROOT/'data/compiler_control8_original_v1.json').read_text())[:1];dataset='compiler_control_q0_original_v2.json';write(ROOT/'data'/dataset,data)
    jobs=[]
    for key in ('teacher','strict'):
        cfg=json.loads((ROOT/'configs'/(item[key]+'.json')).read_text());name=f'V2_StockFA3Probe_Token2_{key}_N1_V8'
        cfg.update(name=name,dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=1,gpus=[0,1],arrival_offsets=[0.],
            source_capsule=new.name,source_capsule_SHA256=digest(new),stock_attention_probe=True,numerical_probe=True,numerical_probe_sync=True)
        if key=='strict':regenerate(cfg,data);suffix=cfg['suffix_by_request']
        else:
            from ..dense_ar_longbench400_pd.geometry import matched_suffixes
            values,_=matched_suffixes([r['input_tokens'] for r in data],2);cfg['suffix_by_request']={f'q{i}':v for i,v in enumerate(values)}
        save(cfg);jobs.append(name)
    write(ROOT/'jobs/stock_attention_diagnostic_v8.json',dict(time=time.time(),lanes=[jobs,[],[],[]],stop_on_failure=True,
        scope='One original q0,identical prefix3763/suffix224 and original serving kernels. '
            'Compare Q,valid KV,Attention output and plan metadata. Keep the failed source5 N8 evidence. '
            'Do not use probe timings or replace strict qualification with this diagnostic.'))
    print('STOCK_SMALL_QUERY_DIAGNOSTIC_FROZEN',digest(new),flush=True)

if __name__=='__main__':main()
