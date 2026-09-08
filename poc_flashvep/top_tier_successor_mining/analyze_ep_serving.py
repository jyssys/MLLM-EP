"""Score direct paired requests, keeping task quality and fixed-length cost apart."""
import argparse
import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--official-source",type=Path,required=True)
    args=ap.parse_args()
    completion=json.loads((args.input/"completed.json").read_text())
    assert completion["exit_codes"]==[0,0],completion
    dataset=completion["arguments"]["dataset"]
    tree=ast.parse(args.official_source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="relaxed_correctness")
    env={"Optional":Optional}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(args.official_source),"exec"),env)
    rows=[json.loads(s) for p in args.input.glob("requests_dp*.jsonl") for s in p.read_text().splitlines()]
    rows=[r for r in rows if not r["warmup"]]
    for r in rows:
        pred=r["prediction"].strip()
        r["correct"]=max(float(env["relaxed_correctness"](a,pred)) if dataset=="chartqa" else
                         float(str(a).strip().lower()==pred.lower()) for a in r["answers"])
        r["output_length"]=len(r["output_tokens"])
        r["dataset"]=dataset
    df=pd.DataFrame(rows)
    keys=["cohort","dp_rank","dataset_request"]
    assert not df.duplicated(keys+["policy"]).any()
    args.output.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output/"requests_scored.csv",index=False)
    base=df[df.policy=="vanilla"]
    summaries=[]
    pair_rows=[]
    for policy,sub in df.groupby("policy"):
        joined=sub.merge(base,on=keys,suffixes=("","_baseline"),validate="one_to_one")
        assert len(joined)==len(sub),"Missing matched vanilla requests"
        joined["e2e_reduction_percent"]=100*(1-joined.e2e_ms/joined.e2e_ms_baseline)
        joined["ttft_reduction_percent"]=100*(1-joined.ttft_ms/joined.ttft_ms_baseline)
        delta=joined.correct-joined.correct_baseline
        # Bootstrap unique images; repeated decoding is not a new quality label.
        image_groups=[part.index.to_numpy() for _,part in joined.groupby("image_id")]
        rng=np.random.default_rng(7317)
        boots=[]
        for _ in range(3000):
            chosen=np.concatenate([image_groups[i] for i in rng.integers(0,len(image_groups),len(image_groups))])
            part=joined.loc[chosen]
            boots.append([part.e2e_reduction_percent.median(),100*(part.correct-part.correct_baseline).mean()])
        boots=np.asarray(boots)
        # Both DP sources share EP collectives. Request/image bootstrap alone
        # understates common serving-state uncertainty within a cohort. Keep it
        # as a secondary diagnostic; primary timing CI resamples paired cohorts.
        cohort_groups=[part.index.to_numpy() for _,part in joined.groupby("cohort")]
        values=joined.e2e_reduction_percent.to_numpy()
        timing_rng=np.random.default_rng(8472)
        timing_boots=[np.median(values[np.concatenate([cohort_groups[i] for i in
            timing_rng.integers(0,len(cohort_groups),len(cohort_groups))])]) for _ in range(3000)]
        wave=[]
        for cohort,w in sub.groupby("cohort"):
            duration=w.completed_monotonic.max()-w.submitted_monotonic.min()
            wave.append({"cohort":cohort,"duration_seconds":duration,"output_tokens":int(w.output_length.sum()),"requests":len(w)})
        duration=sum(w["duration_seconds"] for w in wave)
        itls=np.array([v for values in sub.itls_ms for v in values])
        s={"policy":policy,"dataset":dataset,"observations":len(sub),
           "independent_engine_runs":1,
           "cohort_repetitions":completion["arguments"]["repetitions"],
           "unique_requests":sub.dataset_request.nunique(),"unique_images":sub.image_id.nunique(),
           "accuracy_percent":100*sub.correct.mean(),"paired_quality_delta_pp":100*delta.mean(),
           "quality_delta_95ci_pp":np.quantile(boots[:,1],[.025,.975]).tolist(),
           "e2e_ms_mean":sub.e2e_ms.mean(),"e2e_ms_p50":sub.e2e_ms.median(),
           "e2e_ms_p90":sub.e2e_ms.quantile(.9),"e2e_ms_p99":sub.e2e_ms.quantile(.99),
           "paired_e2e_reduction_median_percent":joined.e2e_reduction_percent.median(),
           "paired_e2e_reduction_95ci_percent":np.quantile(timing_boots,[.025,.975]).tolist(),
           "paired_e2e_ci_unit":"PAIRED_COHORT_BOTH_DP_SOURCES; CONDITIONAL_ON_ONE_ENGINE",
           "paired_timing_cohort_units":len(cohort_groups),
           "paired_e2e_image_bootstrap95ci_percent":np.quantile(boots[:,0],[.025,.975]).tolist(),
           "aggregate_request_latency_reduction_percent":100*(1-joined.e2e_ms.sum()/joined.e2e_ms_baseline.sum()),
           "ttft_ms_p50":sub.ttft_ms.median(),"tpot_ms_p50":sub.tpot_ms.median(),
           "itl_ms_p50":float(np.median(itls)) if len(itls) else None,
           "itl_ms_p99":float(np.quantile(itls,.99)) if len(itls) else None,
           "output_tokens_mean":sub.output_length.mean(),
           "cohort_throughput_output_tokens_s":sum(w["output_tokens"] for w in wave)/duration,
           "cohort_throughput_requests_s":len(sub)/duration,
           "fixed_output":completion["arguments"]["fixed_output"],
           "quality_metric_interpretable":not completion["arguments"]["fixed_output"],
           "itl_exact_request_fraction":float(sub.itl_observations_exact.mean()) if "itl_observations_exact" in sub else None,
           "timing_contract":sub.timing_contract.iloc[0] if "timing_contract" in sub else
                             "HISTORICAL_FRONTEND_RECEIPT_WITHIN_COHORTS",
           "instrumented":completion["arguments"].get("instrument",False),
           "scheduler_context_diagnostic":completion["arguments"].get("scheduler_context",False),
           "batch_label_semantics":"OFFERED_COHORT_CAP_PER_DP_NOT_FIXED_SCHEDULED_BATCH",
           "image_cache_contract":completion["arguments"].get("image_cache","historical_shared"),
           "frontend_e2e_ms_p50":float(sub.frontend_e2e_ms.median()) if "frontend_e2e_ms" in sub else None,
           "frontend_poll_delay_ms_p99":float(sub.frontend_poll_delay_ms.quantile(.99)) if "frontend_poll_delay_ms" in sub else None,
           "paired_frontend_e2e_reduction_percent":float(100*(1-joined.frontend_e2e_ms/joined.frontend_e2e_ms_baseline).median()) if "frontend_e2e_ms" in joined else None,
           "waves":wave}
        summaries.append(s)
        pair_rows.extend(joined[keys+["policy","image_id","e2e_ms","e2e_ms_baseline","e2e_reduction_percent","correct","correct_baseline"]].to_dict("records"))
    pd.DataFrame(pair_rows).to_csv(args.output/"paired_requests.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps(summaries,indent=2,default=lambda x:x.item()))
    print(json.dumps(summaries,indent=2,default=lambda x:x.item()))
    stage_files=list(args.input.glob("stages.pid*.jsonl"))
    if stage_files:
        st=pd.DataFrame([json.loads(s) for p in stage_files for s in p.read_text().splitlines()])
        good=[];bad=[]
        for invocation,g in st.groupby("invocation"):
            if len(g)!=4 or g.ep_rank.nunique()!=4 or g.layer.nunique()!=1:
                bad.append({"invocation":int(invocation),"rows":len(g),"layers":g.layer.tolist()})
                continue
            active=g[g.phase!="dummy"]
            if not len(active):continue
            if active.policy.nunique()!=1 or active.cohort.nunique()!=1:
                bad.append({"invocation":int(invocation),"reason":"POLICY_OR_COHORT_ALIGNMENT"})
                continue
            good.append({"invocation":int(invocation),"layer":int(g.layer.iloc[0]),
                "policy":active.policy.iloc[0],"cohort":active.cohort.iloc[0],
                "phase":"mixed" if active.phase.nunique()>1 else active.phase.iloc[0],
                "critical_moe_ms":g.moe_ms.max(),"critical_dispatch_ms":g.dispatch_ms.max(),
                "critical_expert_ms":g.expert_ms.max(),"critical_combine_ms":g.combine_ms.max(),
                "note":"Maxima may belong to different ranks; do not sum phase maxima."})
        pd.DataFrame(good).to_csv(args.output/"critical_invocations.csv",index=False)
        (args.output/"alignment_validation.json").write_text(json.dumps({"valid_invocations":len(good),"rejected_invocations":len(bad),"rejected_examples":bad[:30]},indent=2))


if __name__=="__main__":main()
