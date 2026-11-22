"""Reset only partial observation buffers with an exact preemption witness."""
import hashlib
import json
import time
import zipfile
from .common import ROOT,write,digest
from .runner import save,signature
from .preemption_repro import replay

def main():
    directory=ROOT/'references/source_capsules';old=directory/'v2_native_decode_scoped_prompt_ops_v4.zip'
    new=directory/'v2_preemption_observer_fix_v5.zip';assert not new.exists()
    with zipfile.ZipFile(old) as archive:files={n:archive.read(n) for n in archive.namelist()}
    name='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py';text=files[name].decode()
    previous=text
    text=text.replace('    native_import_states={};native_export_prefix_states={}',
        '    native_import_states={};native_export_prefix_states={};observer_preemption_resets=[]')
    needle="                    record=pending_hidden.setdefault(external,dict(cursor=0,parts=[]))\n                    assert record['cursor']==start"
    assert text.count(needle)==1
    replacement="""                    record=pending_hidden.setdefault(external,dict(cursor=0,parts=[],last_preemption_serial=-1))
                    if record['cursor']!=start:
                        assert start==0 and record['cursor']>0,('Invalid partial-observer restart',external,start,record['cursor'])
                        witnesses=[(j,p) for j,p in enumerate(preemptions)
                            if j>record['last_preemption_serial'] and p['request_id']==rid
                            and p['computed_before']==record['cursor']]
                        assert witnesses,('Partial-observer restart lacks an exact per-ID preemption witness',rid,record['cursor'])
                        serial,witness=witnesses[-1]
                        observer_preemption_resets.append(dict(request_id=rid,previous_cursor=record['cursor'],
                            witness=witness,discarded_observation_rows=record['cursor']))
                        record=dict(cursor=0,parts=[],last_preemption_serial=serial)
                        pending_hidden[external]=record
                    assert record['cursor']==start"""
    text=text.replace(needle,replacement)
    needle="    Scheduler._preempt_request=preempted\n    import atexit"
    assert text.count(needle)==1
    text=text.replace(needle,needle+"\n    atexit.register(lambda:write(folder/f'partial_observer_preemption_resets_{role}.json',dict(rows=observer_preemption_resets,\n"
        "        scope='Discard only incomplete CPU hidden observations after exact per-ID scheduler-preemption witnesses. No model,KV or scheduler state is changed.')))" )
    assert text!=previous
    result=replay(text);negative=replay(text,False)
    assert result['passed'] and len(result['resets'])==1 and not negative['passed'],(result,negative)
    files[name]=text.encode()
    with zipfile.ZipFile(new,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path,data in sorted(files.items()):
            if path.endswith('.py'):compile(data,path,'exec')
            info=zipfile.ZipInfo(path,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data)
    write(new.with_suffix('.json'),dict(parent_SHA256=digest(old),SHA256=digest(new),changed_files=[name],
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Change the correctness-only partial hidden observer. Preserve every model call,KV tensor,query,scheduler and sampling path. '
            'Keep the continuity assertion. A restart must have an unused exact per-ID preemption event at the old cursor. '
            'Normal recomputation remains visible in query records. Fresh Agg GPU qualification remains required.'))
    write(ROOT/'analysis/agg_partial_preemption_observer_regression_v5.json',dict(positive=result,missing_witness_negative=negative,passed=True))
    original=json.loads((ROOT/'configs/c3_family_qualification_v4_schedule_fix1.json').read_text())
    item=next(x for x in original['items'] if x['label']=='Agg');jobs=[]
    for old_name in item['jobs']:
        cfg=json.loads((ROOT/'configs'/(old_name+'.json')).read_text());job=old_name+'_ObserverFixV5'
        cfg.update(name=job,source_capsule=new.name,source_capsule_SHA256=digest(new));save(cfg);jobs.append(job)
    item=dict(item,jobs=jobs,gate='c3_composition_Agg_v5.json',signature=signature(cfg))
    write(ROOT/'configs/c3_agg_observer_repair_v5.json',dict(time=time.time(),items=[item],state_prefix='c3_agg_observer_repair_v5',
        source_capsule=new.name,source_capsule_SHA256=digest(new),scope='Agg-only observer repair. All prior failed raw files remain immutable. '
            'Other arms retain source4. The changed partial observer is absent when clean performance state_capture=False. '
            'Report the separate capsule hashes. This is not a method-specific scheduling or model optimization.'))
    print('EXACT_PREEMPTION_OBSERVER_REGRESSION_PASS',digest(new),flush=True)

if __name__=='__main__':main()
