"""Validate live route/runtime/output/cache controls before clean comparisons."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
c=json.loads((a.input/'completed.json').read_text());assert c['exit_codes']==[0,0]
proof=[json.loads(line) for f in a.input.glob('runtime_proof*.jsonl') for line in f.read_text().splitlines()]
assert len(proof)==4 and {r['ep_rank'] for r in proof}==set(range(4))
assert all(r['prepare_finalize']=='DeepEPHTPrepareAndFinalize' and r['dp_size']==2 and r['tp_size']==2 and r['ep_size']==4 and not r['dbo'] and r['physical_visible_devices']=='4,5,6,7' for r in proof)
rows=[json.loads(line) for f in a.input.glob('requests_dp*.jsonl') for line in f.read_text().splitlines()]
hashes=[h for r in rows for group in r['mm_hashes'].values() for h in group]
assert len(hashes)==len(set(hashes)),'Cold image identities were reused'
groups={}
for r in rows:
    assert r['completed_monotonic']==r['engine_metrics']['last_token_ts']
    assert r['frontend_completed_monotonic']>=r['completed_monotonic']
    assert r['token_timestamps'][0]==r['engine_metrics']['first_token_ts']
    if not r['warmup']:groups.setdefault((r['cohort'],r['dp_rank'],r['dataset_request']),{})[r['policy']]=r
for group in groups.values():
    assert group['vanilla']['output_tokens']==group['modes_noop']['output_tokens'],'Noop parity'
    assert group['modes_weight_zero']['output_tokens']==group['modes_actual_skip']['output_tokens'],'Sentinel parity'
result=dict(status='PASS',requests=len(rows),matched_requests=len(groups),unique_cold_hashes=len(hashes),
            actual_DeepEP_four_rank_path=True,noop_greedy_equal=True,sentinel_greedy_equal=True,
            primary='SUBMISSION_TO_ENGINE_READY',client_delivery_separately_recorded=True)
(a.input/'RESUME_SANITY_VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
