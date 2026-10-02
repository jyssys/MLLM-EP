import numpy as np
from poc_agentic_ep.contract import lcp
from poc_agentic_ep.metrics import clustered,traffic,choose_replicas


def test_prefix_token_not_string():
    assert lcp([1,2,3],[1,2,4,3])==2
    assert lcp([],[1])==0


def test_shared_prefix_and_phase_manifest():
    import json
    from poc_agentic_ep.contract import ROOT
    d=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text())
    assert len(d['trajectories'])==12 and len(d['rows'])==108
    assert len({r['instance_id'] for r in d['trajectories']})==12
    for r in d['rows']:
        assert r['exclusion_start']>=r['common_prompt_prefix']
        assert r['exclusion_start']>=r['current_observation_frontier']
        assert r['input_token_ids'][:r['prompt_tokens']]==r['prompt_token_ids']
        assert r['input_token_ids'][r['prompt_tokens']:]==r['teacher_forced_ids']
        assert 0<r['new_suffix_tokens']==r['prompt_tokens']-r['exclusion_start']


def test_destination_dedup_not_path_bytes():
    ids=np.broadcast_to(np.arange(32,40,dtype=np.uint8),(48,1,8)).copy()
    src=np.array([0]);pred=np.zeros((48,4,128),dtype=int);pred[:,0,32:40]=1
    before=traffic(ids,src);after=traffic(ids,src,choose_replicas(pred,1))
    assert before['remote_paths']==384 and after['remote_paths']==336
    assert before['remote_dispatch_bytes']==after['remote_dispatch_bytes']==48*4096
    assert traffic(ids,src,choose_replicas(pred,8))['remote_dispatch_bytes']==0


def test_cluster_bootstrap_not_layer_sample():
    result=clustered([0,0,1,1],['A','A','B','B'])
    assert result['trajectories']==2 and result['mean']==.5
    assert result['ci95']==[0.,1.]


def test_idle_sources_no_replication_and_full_tensor_memory():
    pred=np.zeros((48,4,128),dtype=int);pred[:,0,32]=2
    replica=choose_replicas(pred,4)
    assert replica[:,1:].sum()==0
    assert replica.sum()==48
    assert int(replica.sum())*9437184==48*9437184
