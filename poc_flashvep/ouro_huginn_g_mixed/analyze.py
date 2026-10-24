"""Export actual requests and legal plans without treating tokens as independent runs."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .common import ROOT,DEPTHS,write

def csvfile(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/name).open('w') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def main():
    measures=[];requests=[];gaps=[];plans=[];pairs=[];ledger=[];parity=[];decode_ledger=[];selectors=[]
    for path in sorted((ROOT/'raw').glob('*/results.json')):
        job=path.parent.name
        identity=json.loads((path.parent/'identity.json').read_text())
        rows=json.loads(path.read_text())
        reference={r['repeat']:r for r in rows if r['arm']=='mixed'}
        for row in rows:
            base=dict(job=job,gpu=identity['GPU'],model=row['model'],arm=row['arm'],G=row['G'],
                P=row['P'],N=row['N'],C=row['C'],repeat=row['repeat'],lane=row['lane'],evidence='LIVE_MEASURED')
            graph=None
            point={k:v for k,v in row.items() if k not in ['requests','plans','selector_decisions']}
            point['G_semantics']='Unused scheduler knob. Atomic executes full R.' if row['arm']=='atomic' else 'Configured maximum for pure prompt work. Mixed uses G1.'
            point['atomic_actual_G']=DEPTHS[row['model']] if row['arm']=='atomic' else None
            for item in row.get('selector_decisions',[]):selectors.append(dict(base,**item))
            measures.append(dict(base,**point))
            for r in row['requests']:
                ts=r['times'];itl=np.diff(ts)*1000
                requests.append(dict(base,id=r['id'],output_tokens=len(r['tokens']),reason=r['reason'],
                    TTFT_ms=(ts[0]-r['arrival'])*1000,E2E_ms=(ts[-1]-r['arrival'])*1000,
                    ITL_max_ms=float(max(itl)) if len(itl) else None,tokens=r['tokens'],depths=r['depths']))
                for index,gap in enumerate(itl):gaps.append(dict(base,id=r['id'],index=index,ITL_ms=float(gap)))
                assert r['depths']==[DEPTHS[row['model']]]*len(r['tokens'])
            for plan in row['plans']:plans.append(dict(base,**plan))
            prompt_work=defaultdict(lambda:[0]*DEPTHS[row['model']])
            decode_work=defaultdict(lambda:[0]*DEPTHS[row['model']])
            for plan in row['plans']:
                for span in plan['spans']:
                    if plan['operation']=='recurrent' and span['prompt']:
                        for d in range(span['depth'],span['depth']+plan['group']):
                            prompt_work[span['id']][d]+=span['count']
                    elif plan['operation']=='prelude' and plan['group']:
                        for d in range(plan['group']):prompt_work[span['id']][d]+=span['count']
                    elif plan['operation']=='recurrent' and not span['prompt']:
                        for d in range(span['depth'],span['depth']+plan['group']):
                            decode_work[(span['id'],span['start'])][d]+=1
            for rid,work in prompt_work.items():
                expected=row['P'] if rid=='long' else identity['job'].get('background_P',128)
                ledger.append(dict(base,id=rid,actual=work,expected_per_depth=expected,
                    full_depth_pass=all(w==expected for w in work)))
            if row['plans']:
                for request in row['requests']:
                    actual=[v for (rid,pos),v in decode_work.items() if rid==request['id']]
                    decode_ledger.append(dict(base,id=request['id'],
                        decoded_positions=len(actual),expected_positions=len(request['tokens'])-1,
                        full_depth_pass=(len(actual)==len(request['tokens'])-1 and
                            all(v==[1]*DEPTHS[row['model']] for v in actual))))
            ref=reference.get(row['repeat'])
            if ref is not None:
                pairs.append(dict(base,TTFT_ratio_ref_over_candidate=ref['TTFT_long_ms']/row['TTFT_long_ms'],
                    wall_ratio_ref_over_candidate=ref['wall_s']/row['wall_s'],
                    ITL_p99_ratio_ref_over_candidate=(ref['ITL_p99_ms']/row['ITL_p99_ms']) if row['ITL_p99_ms'] else None))
                refs={r['id']:r for r in ref['requests']}
                for r in row['requests']:
                    a=refs[r['id']]
                    first=next((i for i,(x,y) in enumerate(zip(a['tokens'],r['tokens'])) if x!=y),None)
                    parity.append(dict(base,id=r['id'],tokens_exact=a['tokens']==r['tokens'],
                        depths_exact=a['depths']==r['depths'],length_exact=len(a['tokens'])==len(r['tokens']),
                        reason_exact=a['reason']==r['reason'],first_divergent_token=first))
    for name,rows in [('fixed_matrix.csv',measures),('request_metrics.csv',requests),
            ('all_ITL_gaps.csv',gaps),('batch_plan_trace.csv',plans),('paired_cohorts.csv',pairs),
            ('full_depth_ledger.csv',ledger),('actual_output_parity.csv',parity)]:csvfile(name,rows)
    csvfile('decode_full_depth_ledger.csv',decode_ledger)
    csvfile('selector_candidates.csv',selectors)
    write(ROOT/'analysis_counts.json',dict(cohorts=len(measures),requests=len(requests),
        delivered_ITL=len(gaps),plans=len(plans),prompt_ledger=len(ledger),
        prompt_ledger_failures=sum(not r['full_depth_pass'] for r in ledger),
        decode_ledger=len(decode_ledger),decode_ledger_failures=sum(not r['full_depth_pass'] for r in decode_ledger),
        paired_outputs=len(parity),exact_outputs=sum(r['tokens_exact'] for r in parity),
        nonself_paired_outputs=sum(r['arm']!='mixed' for r in parity),
        nonself_exact_outputs=sum(r['tokens_exact'] and r['arm']!='mixed' for r in parity)))
    summary=[]
    grouping=defaultdict(list)
    for row in pairs:
        grouping[(row['model'],row['P'],row['N'],row['C'],row['arm'],row['G'],row['lane'])].append(row)
    rng=np.random.default_rng(20261025)
    for key,rows in grouping.items():
        item=dict(zip(['model','P','N','C','arm','G','lane'],key))
        starts=sorted(set(r['job'] for r in rows));item['process_blocks']=len(starts)
        for metric in ['TTFT_ratio_ref_over_candidate','wall_ratio_ref_over_candidate','ITL_p99_ratio_ref_over_candidate']:
            values=[]
            for s in starts:
                block=[r[metric] for r in rows if r['job']==s and r[metric] is not None]
                if block:values.append(np.mean(block))
            if not values:
                item[metric]=None;item[metric+'_CI95']='NOT_AVAILABLE_no_background_decode';continue
            item[metric]=float(np.mean(values))
            if len(starts)>=3:
                samples=np.array(values)[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
                item[metric+'_CI95']=list(map(float,np.percentile(samples,[2.5,97.5])))
            else:item[metric+'_CI95']='NOT_AVAILABLE_fewer_than_three_process_blocks'
        summary.append(item)
    csvfile('paired_block_summary.csv',summary)
    write(ROOT/'paired_block_summary.json',summary)
    print(json.dumps(json.loads((ROOT/'analysis_counts.json').read_text()),indent=2))

if __name__=='__main__':main()
