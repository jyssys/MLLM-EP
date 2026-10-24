"""Check retained timestamps, full R4 execution, hashes, and real shard ownership."""
import json
import numpy as np
from .common import ROOT,digest,write

def main():
    cohorts=0;requests=0;tokens=0;hashes=0
    for completed in (ROOT/'raw').glob('*/completed.json'):
        data=json.loads(completed.read_text())
        for item in data['artifacts']:
            if item['path']=='telemetry.jsonl':continue
            assert digest(completed.parent/item['path'])==item['sha256'],completed.parent/item['path']
            hashes+=1
        for path in completed.parent.glob('measured_*.json'):
            row=json.loads(path.read_text());cohorts+=1
            gaps=[]
            for r in row['data']['requests']:
                assert len(r['tokens'])==len(r['times'])==len(r['depths'])
                assert all(d==4 for d in r['depths'])
                assert row['data']['start']<=r['times'][0]<=r['times'][-1]<=row['data']['end']
                assert all(b>a for a,b in zip(r['times'],r['times'][1:]))
                gaps.extend((b-a)*1000 for a,b in zip(r['times'],r['times'][1:]))
                requests+=1;tokens+=len(r['tokens'])
            assert abs(np.mean(gaps)-row['ITL_mean_ms'])<1e-9
    operator_cases=0;rank_cases=0
    for group in (ROOT/'distributed').iterdir():
        if not group.is_dir():continue
        identities=list(group.glob('rank*_identity.json'))
        if not identities:continue
        world=json.loads(identities[0].read_text())['world']
        if not all((group/f'rank{r}_completed.json').exists() for r in range(world)):continue
        ranks=[json.loads((group/f'rank{r}_results.json').read_text()) for r in range(world)]
        assert len({len(rs) for rs in ranks})==1
        for rows in zip(*ranks):
            key=(rows[0]['fixture'],rows[0]['B'],rows[0]['placement'],rows[0]['L'])
            assert all((r['fixture'],r['B'],r['placement'],r['L'])==key for r in rows)
            intervals=sorted((r['lo'],r['hi']) for r in rows)
            assert intervals[0][0]==0 and intervals[-1][1]==key[3]
            assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
            assert sum(r['owned_valid_KV_bytes'] for r in rows)==key[1]*key[3]*8192
            for r in rows:
                assert r['GPU_shard_byte_exact']
                for numeric in (r['numerical'],r['graph']['numerical']):
                    assert numeric['finite'] and numeric['relative_L2']<=.03 and numeric['absolute_max']<=.125
            operator_cases+=1;rank_cases+=world
    result=dict(timing_cohorts=cohorts,delivered_requests=requests,delivered_tokens=tokens,
        immutable_worker_artifacts_checked=hashes,operator_cases=operator_cases,rank_cases=rank_cases,
        all_checks_pass=True,scope='Timing and hardware/operator invariants. Not benchmark accuracy.')
    write(ROOT/'verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
