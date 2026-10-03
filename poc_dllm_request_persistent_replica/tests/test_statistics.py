from poc_dllm_request_persistent_replica.statistics import bootstrap_ratio


def test_spec_ratio_of_aggregate_gains():
    r=bootstrap_ratio([1,3],[2,4])
    assert r['mean']==4/6 and r['n']==2
    assert r['ci95'][0]<=r['mean']<=r['ci95'][1]
    assert bootstrap_ratio([0,1],[0,0]) is None
