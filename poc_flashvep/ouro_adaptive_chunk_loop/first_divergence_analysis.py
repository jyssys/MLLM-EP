"""Order captured comparisons without converting symptoms into root causes."""
import json
import re
from .common import ROOT,write

ORDER={'core_input':0,'input_norm':1,'Q_projection':2,'K_projection':3,'V_projection':4,
    'attention':5,'post_attention_norm':6,'MLP':7,'layer_output':8,'core_output':9,'coda_hidden':10,'logits':11}

def order(row):
    key=row['key']
    t=int(re.search(r'^t(\d+)',key)[1])
    d=re.search(r'_d(\d+)',key)
    l=re.search(r'_l(\d+)',key)
    kind=next((name for name in ORDER if key.endswith('_'+name)),None)
    return t,int(d[1]) if d else 5,int(l[1]) if l else (-1 if 'core_input' in key else 25),ORDER.get(kind,99)

def main():
    summaries=[]
    for folder in sorted((ROOT/'raw').glob('p1_*divergence*'))+sorted((ROOT/'raw').glob('p1_common_prefix*')):
        if not (folder/'completed.json').exists():
            continue
        for file in folder.glob('*_vs_*_tensor_comparison.json'):
            rows=json.loads(file.read_text())
            operators=sorted([r for r in rows if r['status']=='COMPARED' and '_KV_' not in r['key'] and not r['key'].endswith('all_target_rows')],key=order)
            mismatches=[r for r in operators if not r['exact']]
            kv=[r for r in rows if '_KV_' in r['key']]
            shape=[r for r in rows if r['status']=='SHAPE_MISMATCH']
            summaries.append(dict(run_id=folder.name,comparison=file.name,
                earliest_captured_difference=mismatches[0] if mismatches else None,
                captured_operator_values=len(operators),captured_nonexact_values=len(mismatches),
                KV_names_compared=len(kv),KV_nonexact=sum(not r.get('exact',False) for r in kv),
                shape_mismatch_keys=[r['key'] for r in shape],scope='Earliest captured point, not necessarily first causal operation.',
                classification='UNKNOWN_UNTIL_MATCHED_STATE_AND_BATCH_GEOMETRY_REPLAY'))
    write(ROOT/'first_divergence_summary.json',summaries)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    main()
