"""One unchanged inherited grader for every arm. No GPU execution."""
import csv
import numpy as np
from poc_async_recurrent_pp.scoring import score
from .contract import *

def main():
    byid={r['id']:r for r in dataset(False)};rows=[]
    cache_path=ROOT/'state/grading_cache.json'
    cache=read(cache_path) if cache_path.exists() else {}
    grader_identity=sha(PROJECT/'poc_async_recurrent_pp/scoring.py')
    for folder in sorted((ROOT/'raw').iterdir()):
        for file in sorted(folder.glob('*.json')):
            try:r=read(file)
            except json.JSONDecodeError:continue
            if not isinstance(r,dict) or r.get('id') not in byid or 'output_ids' not in r:continue
            if r.get('evaluation_eligible') is False:continue
            cache_key=hashlib.sha256(json.dumps([grader_identity,r['id'],byid[r['id']]['answer'],r['text'],r['finish_reason']],sort_keys=True).encode()).hexdigest()
            if cache_key not in cache:
                cache[cache_key]=score(byid[r['id']],dict(text=r['text'],finish_reason='eos' if r['finish_reason'] in ['eos','stop'] else 'length'))
            parsed=cache[cache_key]
            row=dict(run=folder.name,id=r['id'],dataset=r['dataset'],arm=r['arm'],correct=parsed['answer_matches'],
                prompt_ID_parity=r['prompt_ids']==byid[r['id']].get('input_ids',r['prompt_ids']),
                parsed=parsed['parsed'],answer=parsed['extracted_answer'],gold=byid[r['id']]['answer'],
                output_tokens=r['output_tokens'],finish_reason=r['finish_reason'],mean_depth=r['mean_depth'],
                decode_tokens=max(0,r['output_tokens']-1),
                decode_loop_work=sum(r['depths'][1:]) if 'depths' in r else 4*max(0,r['output_tokens']-1),
                decode_R2=sum(d==2 for d in r.get('depths',[])[1:]),decode_R3=sum(d==3 for d in r.get('depths',[])[1:]),
                decode_R4=sum(d==4 for d in r['depths'][1:]) if 'depths' in r else max(0,r['output_tokens']-1),
                E2E_s=r['E2E_s'],status='FAILED_INPUT_ID_PARITY' if r['prompt_ids']!=byid[r['id']].get('input_ids',r['prompt_ids']) else r['status'],
                source=str(file.relative_to(ROOT)))
            rows.append(row)
    write(cache_path,cache)
    from .effective_sources import sources,subset_views
    effective=sources();write(ROOT/'effective_source_manifest.json',effective)
    for group in effective['cohorts']:
        for source in group['question_files']:
            match=next((r for r in rows if r['source']==source),None)
            if match:rows.append(dict(match,run='quality_S4_canonical',source_run=group['source_run']))
    views=subset_views();write(ROOT/'subset400_source_manifest.json',views)
    actual={r['source']:r for r in rows if not r.get('source_run')}
    for view in views.values():
        for group in view['cohorts']:
            for source in group['question_files']:
                if source in actual:rows.append(dict(actual[source],run=view['name'],source_run=group['source_run']))
    with (ROOT/'evaluation_outputs.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    summary=[]
    for run in sorted({r['run'] for r in rows}):
        for domain in ['gsm8k','math500']:
            selected=[r for r in rows if r['run']==run and r['dataset']==domain]
            if not selected:continue
            n=len(selected);correct=sum(r['correct'] for r in selected)
            summary.append(dict(run=run,arm=selected[0]['arm'],dataset=domain,N=n,correct=correct,accuracy_percent=100*correct/n,
                invalid_prompt_ID_records=sum(not r['prompt_ID_parity'] for r in selected),
                completed_correct=sum(r['correct'] and r['finish_reason']!='length' for r in selected),
                completed_correct_percent_of_all=100*sum(r['correct'] and r['finish_reason']!='length' for r in selected)/n,
                parse_failures=sum(not r['parsed'] for r in selected),truncated=sum(r['finish_reason']=='length' for r in selected),
                avg_output_tokens=float(np.mean([r['output_tokens'] for r in selected])),mean_depth=float(np.mean([r['mean_depth'] for r in selected])),
                total_output_tokens=sum(r['output_tokens'] for r in selected),total_decode_tokens=sum(r['decode_tokens'] for r in selected),
                total_decode_loop_work=sum(r['decode_loop_work'] for r in selected),
                mean_decode_depth=sum(r['decode_loop_work'] for r in selected)/sum(r['decode_tokens'] for r in selected) if sum(r['decode_tokens'] for r in selected) else None,
                decode_R2=sum(r['decode_R2'] for r in selected),decode_R3=sum(r['decode_R3'] for r in selected),decode_R4=sum(r['decode_R4'] for r in selected),
                status='FAILED_INPUT_ID_PARITY' if any(not r['prompt_ID_parity'] for r in selected) else selected[0]['status']))
    if summary:
        with (ROOT/'accuracy_by_arm.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
    write(ROOT/'state/scoring_progress.json',summary)
    print('SCORED',len({r['source'] for r in rows}),'unique actual-generation records;',len(rows),'rows including checkpoint views',flush=True)

if __name__=='__main__':main()
