"""Native full-layer paired timings, explicitly not full VL request E2E."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    assert not (args.input/'INVALIDATED.json').exists(), 'Invalidated native run is not clean evidence'
    completed=[json.loads(p.read_text()) for p in args.input.glob("completed_*.json")]
    assert len(completed)==4 and all(r["layers"]==48 and not r["dummy"] for r in completed)
    rows=[json.loads(line) for p in args.input.glob("rank*.jsonl") for line in p.read_text().splitlines()]
    df=pd.DataFrame(rows)
    df=df[df["repeat"]>=0].copy()
    assert not df.concurrent_other_workload.any(),"Concurrent runs are functional only"
    assert not df.get('profiled_diagnostic',pd.Series(False,index=df.index)).any(), "Profiled timelines are not clean paired speed evidence"
    keys=["rank","case","repeat"]
    assert not df.duplicated(keys+["policy"]).any()
    args.output.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output/"rank_prefill.csv",index=False)
    base=df[df.policy=="vanilla"]
    libra=df[df.policy=="libra"]
    result=[]
    for policy,g in df.groupby("policy"):
        pair=g.merge(base,on=keys,suffixes=("","_baseline"),validate="one_to_one")
        assert len(pair)==len(g)
        delta=100*(1-pair.prefill_ms/pair.prefill_ms_baseline)
        # Preserve coupled four-rank executions in bootstrap resampling.
        units=[part.index.to_numpy() for _,part in pair.groupby(["case","repeat"])]
        rng=np.random.default_rng(8173)
        boots=[]
        for _ in range(3000):
            chosen=np.concatenate([units[i] for i in rng.integers(0,len(units),len(units))])
            boots.append(float(delta.iloc[chosen].median()))
        row={"policy":policy,"rank_observations":len(g),"paired_execution_units":len(units),
             "independent_engine_runs":1,"scope":g.scope.unique().tolist(),
             "prefill_ms_p50":float(g.prefill_ms.median()),"prefill_ms_p90":float(g.prefill_ms.quantile(.9)),
             "prefill_ms_mean":float(g.prefill_ms.mean()),"prefill_ms_cv":float(g.prefill_ms.std()/g.prefill_ms.mean()),
             "paired_reduction_vs_vanilla_median_percent":float(delta.median()),
             "paired_unit_bootstrap95ci_percent":np.quantile(boots,[.025,.975]).tolist(),
             "greedy_first_equal_fraction":float(g.greedy_first_equal.mean()),
             "min_logit_cosine_vs_vanilla":float(g.logit_cosine.min()),
             "max_relative_l2_vs_vanilla":float(g.relative_logit_l2.max()),
             "max_logit_abs_vs_vanilla":float(g.logit_max_abs.max()),
             "full_request_e2e_gain":None}
        critical=g.groupby(['case','repeat']).prefill_ms.max()
        critical_base=base.groupby(['case','repeat']).prefill_ms.max()
        critical_delta=100*(1-critical/critical_base)
        critical_rng=np.random.default_rng(9517)
        critical_values=critical_delta.to_numpy()
        critical_boots=np.median(critical_values[critical_rng.integers(
            0,len(critical_values),(3000,len(critical_values)))],axis=1)
        row.update(rank_critical_prefill_ms_p50=float(critical.median()),
            rank_critical_paired_reduction_vs_vanilla_percent=float(critical_delta.median()),
            rank_critical_reduction_95ci_percent=np.quantile(critical_boots,[.025,.975]).tolist())
        if "hf_reference_greedy_equal" in g:
            row.update(hf_reference_greedy_equal_fraction=float(g.hf_reference_greedy_equal.mean()),
                       hf_reference_min_cosine=float(g.hf_reference_cosine.min()),
                       hf_reference_max_relative_l2=float(g.hf_reference_relative_l2.max()))
        if 'hf_reference_kl_to_policy' in g:
            row.update(hf_reference_kl_p50=float(g.hf_reference_kl_to_policy.median()),
                       hf_reference_kl_max=float(g.hf_reference_kl_to_policy.max()))
        if 'logit_kl_vanilla_to_policy' in g:
            row.update(kl_vs_vanilla_p50=float(g.logit_kl_vanilla_to_policy.median()),
                       kl_vs_vanilla_max=float(g.logit_kl_vanilla_to_policy.max()))
        if policy in ("libra_oracle","libra_frozen_oracle"):
            comparison=libra if policy=="libra_oracle" else df[df.policy=="libra_frozen"]
            oracle_pair=g.merge(comparison,on=keys,suffixes=("","_libra"),validate="one_to_one")
            row["oracle_comparison_policy"]=comparison.policy.iloc[0]
            row["controlled_replay_not_model_correctness"]=policy=="libra_frozen_oracle"
            row["oracle_paired_prefill_gain_vs_libra_percent"]=float((100*(1-oracle_pair.prefill_ms/oracle_pair.prefill_ms_libra)).median())
            comparison_critical=comparison.groupby(['case','repeat']).prefill_ms.max()
            oracle_critical_delta=100*(1-critical/comparison_critical)
            row['oracle_rank_critical_gain_percent']=float(oracle_critical_delta.median())
            values=oracle_critical_delta.to_numpy()
            boots=np.median(values[critical_rng.integers(0,len(values),(3000,len(values)))],axis=1)
            row['oracle_rank_critical_gain_95ci_percent']=np.quantile(boots,[.025,.975]).tolist()
            row["oracle_realized_topk_recall_mean"]=float(g.oracle_realized_topk_recall_mean.mean())
            row["oracle_realized_topk_recall_min"]=float(g.oracle_realized_topk_recall_min.min())
        result.append(row)
        pair.to_csv(args.output/f"paired_{policy}.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
