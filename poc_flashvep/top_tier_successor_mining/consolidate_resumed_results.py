"""Consolidate completed fresh GPU evidence without manufacturing absent results."""
import argparse
import json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
r=a.result;out=r/'analysis/resumed_consolidated_20260908';out.mkdir(parents=True,exist_ok=True)
ep=[];libra=[]
for folder in sorted((r/'analysis').glob('ep_*20260908')):
    raw=r/'online'/folder.name
    if not (folder/'summary.json').exists() or not (raw/'completed.json').exists():continue
    if (raw/'INVALIDATED.json').exists() or 'sanity' in folder.name:continue
    comp=json.loads((raw/'completed.json').read_text())
    if comp.get('exit_codes')!=[0,0]:continue
    rows=json.loads((folder/'summary.json').read_text())
    if not isinstance(rows,list):continue
    for row in rows:
        ep.append({**{k:v for k,v in row.items() if k!='waves'},'run':folder.name,
            'batch_per_dp':comp['arguments']['batch_per_dp'],'request_cap':comp['arguments']['requests'],
            'status':'DIAGNOSTIC' if row['instrumented'] or row['fixed_output'] else 'CLEAN_REQUEST_TIMING'})
for folder in sorted((r/'analysis').glob('libra_native*')):
    raw=r/'online'/folder.name
    if not (folder/'summary.json').exists():continue
    if (raw/'INVALIDATED.json').exists():continue
    if 'layoutfix' in folder.name or 'hidden_debug' in folder.name:continue
    if not list(raw.glob('completed_*.json')):continue
    args=json.loads((raw/'arguments.json').read_text())
    if args.get('profiled')=='True':continue
    for row in json.loads((folder/'summary.json').read_text()):
        libra.append(dict(row,run=folder.name,tokens_per_source=int(args['tokens']),
            context='VL_POST_VISION_PREFILL' if args.get('vl_inputs')!='None' else 'TEXT_PREFILL'))
pd.DataFrame(ep).to_csv(out/'ep_all_conditions.csv',index=False)
pd.DataFrame(libra).to_csv(out/'libra_all_conditions.csv',index=False)
result=dict(ep_condition_policy_rows=len(ep),libra_condition_policy_rows=len(libra),
    clean_ep_engine_runs=len({x['run'] for x in ep if x['status']=='CLEAN_REQUEST_TIMING'}),
    warnings=['Cohort repetitions are not independent engine restarts.',
              'Libra captured-VL measurements exclude vision encoding and are not request E2E.',
              'Fixed-current-route oracle rows are causal replay, not model-quality evidence.',
              'No Kimi/general MLLM or original-paper speedup reproduction claim.'])
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
