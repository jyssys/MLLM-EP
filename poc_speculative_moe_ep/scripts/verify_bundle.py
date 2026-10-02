"""Read-only integrity and recomputation checks; does not initialize CUDA."""
import argparse
import json

import numpy as np
import pandas as pd

from poc_speculative_moe_ep.contract import ROOT,sha
from poc_speculative_moe_ep.analysis import route_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--trace-version',default='phase_a_v1');a=p.parse_args()
    root=ROOT/'artifacts'/a.evidence;manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest['exports']:
        assert sha(root/row['path'])==row['sha256'],row['path']
    truth=pd.read_parquet(ROOT/'artifacts'/a.trace_version/'route_metrics.parquet')
    verifies=pd.read_parquet(ROOT/'artifacts'/a.trace_version/'verification_metrics.parquet')
    checked=0
    for index in sorted(root.glob('*/route_index.json')):
        run=index.parent.name;meta=json.loads(index.read_text())
        owner=[]
        for rank in range(4):
            proof=json.loads((index.parent/f'ownership.rank{rank}.json').read_text())
            expected=[-1]*128;expected[rank*32:(rank+1)*32]=list(range(32))
            assert proof['expert_map']==expected
            assert proof['worker_cuda_visible_devices']=='2,3,6,7'
            if proof.get('actual_cuda_uuid') not in (None,'UNAVAILABLE'):
                env_path=ROOT/'artifacts/environment.json'
                if not env_path.exists():env_path=ROOT/'artifacts/environment_public.json'
                env=json.loads(env_path.read_text())
                expected_uuid=env['physical_logical_mapping'][rank]['nvidia_smi'].split(',')[1].strip().removeprefix('GPU-')
                assert proof['actual_cuda_uuid'].removeprefix('GPU-')==expected_uuid
            owner.extend(range(rank*32,(rank+1)*32))
        assert sorted(owner)==list(range(128))
        with np.load(index.parent/'packed_routes.npz') as packed:
            for wave in meta['waves']:
                # One whole48-layer invocation per request recomputed from raw
                # ID arrays, independent of the stored derived counters.
                forward=wave['forwards'][0]
                matching=verifies[(verifies.run==run)&(verifies.wave==wave['wave'])&(verifies.forward==forward)]
                accepted=int(matching.iloc[0].raw_accepted_draft_tokens) if len(matching) else wave['accepted'][0]
                for layer in range(48):
                    ids=[packed[r['key']+'_ids'][0,layer,:r['source_rows'][0]] for r in wave['ranks']]
                    for r in wave['ranks']:
                        weights=packed[r['key']+'_weights'][0,layer,:r['source_rows'][0]]
                        assert np.all(weights>=0)
                        assert np.allclose(weights.sum(axis=1),1.,atol=2e-3), 'selected route renormalization changed'
                    counts=[r['valid_source_rows'][0] for r in wave['ranks']]
                    value=route_features(ids,counts,accepted)
                    assert value['valid_paths']==8*value['token_rows']
                    assert value['logical_dispatch_bytes']==value['logical_combine_bytes']
                    if len(matching):
                        row=truth[(truth.run==run)&(truth.wave==wave['wave'])&(truth.forward==forward)&(truth.layer==layer)].iloc[0]
                        for key in ('valid_paths','executed_paths','unique_experts','max_rank_bytes','logical_dispatch_bytes','remote_routed_paths','rejected_logical_dispatch_bytes'):
                            assert value[key]==row[key],(run,wave['wave'],layer,key)
                    checked+=1
    print(json.dumps(dict(status='PASS',hashes=len(manifest['exports']),recomputed_layer_events=checked)))


if __name__=='__main__':main()
