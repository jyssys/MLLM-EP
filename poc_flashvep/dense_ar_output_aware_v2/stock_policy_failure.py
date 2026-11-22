"""Audit a failed Stock Token state gate without changing its verdict."""
import collections
import json
from pathlib import Path
import re
import time
import torch
from .common import ROOT,write,digest

def run():
    torch.set_num_threads(1)
    passed='V2_C3Policy_StockToken_Cap2_Credit8_Strict8'
    failed='V2_C3Policy_StockToken_Cap2_Credit16_Strict8'
    names=(passed,failed);cfg=[json.loads((ROOT/'configs'/(n+'.json')).read_text()) for n in names]
    differences={k:[c.get(k) for c in cfg] for k in set().union(*(c.keys() for c in cfg)) if cfg[0].get(k)!=cfg[1].get(k)}
    assert set(differences)=={'name','gpus'},differences
    assert all(c['continuous_source_credit_limit']==0 for c in cfg)
    states={};graphs={};geometry={}
    for role in ('prefill','decode'):
        states[role]=[json.loads((ROOT/'raw'/n/f'axis_events_{role}.json').read_text())['states'] for n in names]
        q=[];g=[]
        for n in names:
            folder=ROOT/'raw'/n
            rows=json.loads((folder/f'actual_queries_{role}.json').read_text())['rows']
            q.append(collections.Counter((r['request_id'],r['start'],r['count'],r['actual_batch_queries'],r['requests'])
                for r in rows if r['start']<r['original_input_tokens']))
            text=(folder/f'{role}.log').read_text();cache=Path(re.search(r'Using cache directory: (.+?) for vLLM',text).group(1))
            fx=cache/'computation_graph.py';g.append(dict(cache=str(cache),FX_SHA256=digest(fx)))
        geometry[role]=dict(valid_prompt_query_geometry_equal=q[0]==q[1],query_records=[sum(v.values()) for v in q])
        graphs[role]=dict(rows=g,FX_graph_equal=g[0]['FX_SHA256']==g[1]['FX_SHA256'])
    rows=[]
    for i in range(8):
        rid='q'+str(i);a,b=states['decode'][0][rid],states['decode'][1][rid]
        ah=torch.load(a['tensor_file'],map_location='cpu',weights_only=False)['hidden'][-1].float()
        bh=torch.load(b['tensor_file'],map_location='cpu',weights_only=False)['hidden'][-1].float()
        delta=(ah-bh).abs();first=next((j for j,(x,y) in enumerate(zip(a['KV'],b['KV'])) if x!=y),None)
        rows.append(dict(request_id=rid,full_P_prefix_KV_exact=states['prefill'][0][rid]['KV']==states['prefill'][1][rid]['KV'],
            first_different_D_full_KV_layer=first,last_hidden_max_abs=float(delta.max()),
            last_hidden_RMS=float(delta.square().mean().sqrt()),
            last_hidden_relative_RMS=float(delta.square().mean().sqrt()/ah.square().mean().sqrt()),
            evidence='DIAGNOSTIC_TENSOR_COMPARISON'))
    result=dict(time=time.time(),passed_control=passed,failed_gate=failed,configuration_differences=differences,
        prompt_geometry=geometry,compiled_FX_graphs=graphs,rows=rows,
        status='STRICT_STATE_MISMATCH_ROOT_CAUSE_NOT_IDENTIFIED',failed_gate_remains_failed=True,
        scope='The serial strict probes both set coordinator credit to0. Their names and GPU pairs differ. '
            'Do not attribute this mismatch to source credit16. Compare full-prefix hashes and retained final-hidden tensors only. '
            'A small sampled tensor difference or unchanged greedy output does not establish full-suffix numerical tolerance. '
            'No new GPU run,model correction,threshold change or promotion of the failed configuration occurs.')
    write(ROOT/'analysis/stock_policy_strict_failure_audit_v13.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
