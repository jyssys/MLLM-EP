"""Recompute every route count, policy decision and causal ordering offline."""
import argparse
import gzip
import json
import numpy as np
from poc_agentic_ep.contract import ROOT,sha,write
from poc_agentic_ep.metrics import source_counts,choose_replicas


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args();root=ROOT/'results'/a.version
    exported=json.loads((root/'compact_manifest.json').read_text())
    for name,digest in exported['files'].items():assert sha(root/name)==digest,name
    environment=json.loads((ROOT/'results/environment.json').read_text())
    gpu_map={int(line.split(',')[0]):line.split(',')[1].strip().removeprefix('GPU-') for line in environment['gpu_map'].splitlines()}
    for rank,physical in enumerate((2,3,6,7)):
        proof=json.loads((root/f'ownership.rank{rank}.json').read_text())
        assert proof['actual_cuda_uuid'].removeprefix('GPU-')==gpu_map[physical]
        assert proof['worker_cuda_visible_devices']=='2,3,6,7'
    frozen=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text())
    look={r['request_id']:r for r in frozen['rows']}
    agg=np.load(root/'aggregated.npz');ids=agg['request_ids'].tolist();counts=agg['source_expert_counts'];index={rid:i for i,rid in enumerate(ids)}
    destinations=np.load(root/'destination_ep_ranks.npz')
    layers=0
    for rid in ids:
        with np.load(root/'raw_routing'/(rid+'.npz')) as raw:
            assert raw['expert_ids'].shape[0]==48 and raw['expert_ids'].shape[-1]==8
            assert np.array_equal(raw['expert_ids']//32,destinations[rid])
            assert np.allclose(raw['routing_weights'].sum(axis=-1),1,atol=2e-3)
            assert (raw['positions']>=look[rid]['exclusion_start']).all()
            assert np.array_equal(raw['token_category']==1,raw['positions']>=look[rid]['prompt_tokens'])
            for phase in range(2):
                mask=raw['token_category']==phase
                computed=source_counts(raw['expert_ids'][:,mask,:],raw['source_ep_rank'][mask])
                assert np.array_equal(computed,counts[index[rid],phase])
                assert computed.sum()==48*8*int(mask.sum())
                layers+=48
    records=json.loads((root/'native_worker_records.json').read_text())['rows']
    waves={r['request_id']:r['wave'] for r in records if r['split']!='diagnostic'}
    calib=[index[rid] for rid in ids if look[rid]['split']=='calibration'];global_counts=counts[calib].sum(axis=0)
    assert max(waves[ids[i]] for i in calib)<min(waves[rid] for rid in ids if look[rid]['split']=='heldout')
    policies=json.load(gzip.open(root/'oracle_policy_assignments.json.gz','rt'))['rows']
    for policy in policies:
        target=policy['target'];previous=policy['previous'];phase=0 if policy['phase']=='new_prefill' else 1
        assert look[target]['trajectory_id']==look[previous]['trajectory_id']
        assert look[target]['turn']==look[previous]['turn']+1 and waves[previous]<waves[target]
        if policy['policy']=='global':pred=global_counts[phase]
        elif policy['policy']=='previous':pred=counts[index[previous],phase]
        else:assert policy['policy']=='oracle';pred=counts[index[target],phase]
        assert policy['source_truth_used']==(policy['policy']=='oracle')
        assert np.array_equal(np.argwhere(choose_replicas(pred,policy['slots'])),np.asarray(policy['replica_triplets']))
    result=dict(status='PASS',hashes=len(exported['files']),turns=len(ids),recomputed_layer_phase_events=layers,
        policy_decisions_recomputed=len(policies),calibration_completed_before_heldout=True,
        previous_turn_strictly_before_target=True,only_explicit_oracle_uses_next_truth=True,
        shared_prefix_in_primary=False,independent_bootstrap_unit='trajectory')
    output=root/'no_future_leakage_and_integrity_audit.json'
    if not output.exists():write(output,result)
    else:assert json.loads(output.read_text())==result
    print(json.dumps(result))


if __name__=='__main__':main()
