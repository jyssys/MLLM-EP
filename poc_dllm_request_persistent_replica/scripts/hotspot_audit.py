"""Expose absolute counts behind worst relative hotspots; no policy retuning."""
import json
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *
from poc_dllm_request_persistent_replica.data import load_profile


def main():
    summary=json.loads(summary_path().read_text());c=summary['selected']
    raw=pd.read_json(ROOT/'results/simulation_long_v2/request_policy.jsonl',lines=True)
    manifest=json.loads((ROOT/'results/long_manifest.json').read_text())
    rows=[]
    for info in manifest['requests']:
        if info['split']!='heldout':continue
        profile=load_profile(info)
        for policy in (c['policy'],c['global_control']):
            r=raw[(raw.request==info['request'])&(raw.policy==policy)&
                (raw.budget_gib==c['budget_gib'])&(raw.load_guard==c['load_guard'])].iloc[0]
            selected={int(s):{int(l):set(es) for l,es in ls.items()} for s,ls in r['plan'].items()}
            result=profile.evaluate(selected,int(r['profile_blocks']))
            assert result['post_bytes']==r['post_bytes']
            rows.append(dict(request=info['request'],policy=policy,**result['worst_stage_detail'],
                worst_ratio_increase=result['worst_stage_load_increase'],
                cumulative_max_increase=result['max_rank_load_increase'],
                summed_stage_critical_increase=result['stage_critical_load_increase']))
    pd.DataFrame(rows).to_csv(ROOT/'results/absolute_hotspot_counts.csv',index=False)
    new_json(ROOT/'results/hotspot_interpretation.json',dict(policy_unchanged=True,
        caveat='Large worst-stage relative ratios may occur with few residual live paths. They are a failed predeclared load gate, NOT measured straggler latency. Cumulative/summed-critical counts are reported separately.',
        stage_b_not_rescued=True,actual_GPU_time_not_measured=True))


if __name__=='__main__':main()
