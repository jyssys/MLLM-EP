"""Request/wave metrics. Clean serving and trace components stay distinct."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from poc_native_lp_ep4.contract import atomic_json,read_json


def victim_itls(wave,decode_tokens,with_ends=False):
    emitted=wave["emitted"]
    resident={rid for rid,tokens in emitted.items() if len(tokens)==decode_tokens}
    times=defaultdict(list)
    for step in wave["steps"]:
        for out in step["outputs"]:
            if out["id"] in resident:
                times[out["id"]].extend([step.get("engine_end_ns",step["end_ns"])]*len(out["tokens"]))
    # Order by request admission number; each DP uses the same resident order.
    # Caller IDs gain an internal random suffix, but DELTA outputs retain the
    # numeric admission ID. Lexicographic ordering (9 after 12) would silently
    # mismatch the four canonical resident prompts between paired waves.
    order=[str(rid).split('-')[0] for rid in wave['resident_ids']]
    assert set(order)==set(times), (order,list(times))
    arrays=[np.diff(times[rid])/1e6 for rid in order]
    ends=[np.array(times[rid][1:]) for rid in order]
    return (arrays,ends) if with_ends else arrays


def clean_rows(run):
    manifest=read_json(run/"manifest.json")
    decoded=manifest["decode_tokens"]
    data=[read_json(run/f"worker.dp{dp}.json")["waves"] for dp in (0,1)]
    lm_entries={e['wave']:e['entry_ns'] for path in run.glob('model_entry*.jsonl')
                for e in (json.loads(line) for line in path.open()) if e['dp_rank']==0}
    baseline={w["repeat"]:w for w in data[0] if w["arm"]=="DECODE_ONLY" and w["repeat"]>=0}
    rows=[]
    for wave in data[0]:
        if wave["repeat"]<0:continue
        partner=next(w for w in data[1] if w["wave"]==wave["wave"])
        a0,e0=victim_itls(wave,decoded,True);a1,e1=victim_itls(partner,decoded,True)
        arrays=a0+a1;ends=e0+e1
        # Steady resident decode starts before injection. Never include the
        # initial image-prefill / first-token interval in TBT statistics.
        start=max(1,manifest["inject_at"]-4)
        flat=np.concatenate([a[start:] for a in arrays])
        base=(next(w for w in data[0] if w['wave']==wave['baseline_wave'])
              if wave.get('baseline_wave') is not None else baseline.get(wave["repeat"]))
        if wave['arm']=='DECODE_ONLY':base=wave
        excess=[]
        window=[]
        if base is not None:
            bp=next(w for w in data[1] if w["wave"]==base["wave"])
            ba=victim_itls(base,decoded)+victim_itls(bp,decoded)
            radius=2*np.median(np.concatenate(ba))*1e6
            lo=wave["injection_begin_ns"];hi=wave.get('injection_engine_finish_ns',wave["injection_finish_ns"])
            for a,b,t in zip(arrays,ba,ends):
                mask=np.arange(len(a))>=start
                if lo is not None and hi is not None: mask &= (t>=lo)&(t<=hi+radius)
                excess.extend((a-b)[mask].tolist())
                window.extend(a[mask].tolist())
        peak=float(max(window)) if window else float(flat.max())
        rows.append({"run":str(run),"arm":wave["arm"],"modality":wave["modality"],
            "repeat":wave["repeat"],"wave":wave["wave"],"ttft_ms":wave["ttft_ms"],
            'lm_start_to_first_token_ms':None if wave['wave'] not in lm_entries or wave.get('injection_engine_finish_ns') is None
                else (wave['injection_engine_finish_ns']-lm_entries[wave['wave']])/1e6,
            "mean_itl_ms":float(flat.mean()),"p95_itl_ms":float(np.percentile(flat,95)),
            "p99_itl_ms":float(np.percentile(flat,99)),"peak_itl_ms":peak,
            "peak_vsel_ms":None if not excess else float(max(excess)),
            "prompt_tokens":wave["prompt_tokens"],"resident_parity":wave["resident_parity"],
            "injection_enqueue_ms":wave.get('injection_enqueue_ms'),
            "token_timestamp":"ENGINE_COMPLETION" if 'engine_end_ns' in wave['steps'][0] else 'CLIENT_POLL_DIAGNOSTIC_ONLY',
            "paired_baseline_wave":None if base is None else base['wave'],
            "paired_baseline_median_itl_ms":None if base is None else float(np.median(np.concatenate(ba))),
            "peak_itl_ratio":None if base is None else peak/float(np.median(np.concatenate(ba))),
            "slo_violation_fraction":None if base is None else float((flat>1.5*np.median(np.concatenate(ba))).mean()),
            "instrumented":wave.get('instrumented',manifest['trace']),
            "scope":"CONTROLLED_NATIVE_LP_EP4_HARNESS",
            "lane":"TRACE_TIMING_NOT_HEADLINE" if wave.get('instrumented',manifest['trace']) else "HOOK_BYPASS_CONTROL" if manifest['trace'] else "CLEAN"})
    return rows


def trace_rows(run):
    events=[json.loads(l) for p in run.glob("stage*.jsonl") for l in p.open()]
    manifests=read_json(run/"manifest.json")
    waves=manifests["waves"]
    groups=defaultdict(list)
    for e in events:
        if e.get("replicate",-1)>=0:
            if 'global_prefill_rows' in e:
                frame=('lp_stage',e['native_lp_stage']) if e.get('native_lp_stage') is not None else (
                    ('prefill_offset',e['prefill_offset']) if e['global_prefill_rows'] else ('decode_step',e['global_decode_step']))
            else:frame=('legacy_internal',e.get('iteration',e.get('outer_iteration')))
            groups[(e["wave"],frame,e["layer"])].append(e)
    rows=[]
    for key,es in groups.items():
        w=waves[key[0]]
        if w["arm"].startswith("LP_"):
            prefill=any(e.get("native_lp_active_prefill") is True for e in es)
        else:
            prefill=any(e.get('global_prefill_rows',e.get('injected_rows',e.get('route_stats',{}).get('injected',{}).get('tokens',0)))>0 for e in es)
        fields=["dispatch_ms","expert_execution_ms","combine_ms","moe_ms"]
        row={"run":str(run),"wave":key[0],"iteration":key[1],"layer":key[2],
            "arm":w["arm"],"modality":w["modality"],"repeat":w["repeat"],
            "prefill_active":prefill,"ranks":len(es),
            'outer_iteration':es[0].get('outer_iteration')}
        row['global_decode_step']=es[0].get('global_decode_step')
        # Durations are local CUDA elapsed times, never raw timestamps aligned
        # across devices. Keep critical-duration proxies and per-rank data.
        for field in fields: row[field]=float(max(e[field] for e in es))
        row["rank_durations"]={str(e["ep_rank"]):{f:e[f] for f in fields} for e in es}
        row["loaded_expert_weight_bytes_per_expert"]=next((e.get("expert_weight_bytes_per_expert") for e in es if "expert_weight_bytes_per_expert" in e),None)
        row["local_active_experts"]={str(e["ep_rank"]):e.get("local_distinct_active_experts") for e in es}
        row["local_pairs"]={str(e["ep_rank"]):e.get("local_token_expert_pairs") for e in es}
        routes=[e["route_stats"].get("injected",{}) for e in es if "route_stats" in e]
        hist=np.zeros(128,dtype=int)
        bytes_dest=np.zeros(4,dtype=int)
        traffic=np.zeros((4,4),dtype=np.int64)
        for r in routes:
            if "expert_histogram" in r:hist+=np.array(r["expert_histogram"])
            if "dispatch_payload_bytes_per_rank" in r:bytes_dest+=np.array(r["dispatch_payload_bytes_per_rank"])
        for e in es:
            r=e.get('route_stats',{}).get('injected',{})
            if 'dispatch_payload_bytes_per_rank' in r:
                traffic[int(e['ep_rank'])]+=np.array(r['dispatch_payload_bytes_per_rank'],dtype=np.int64)
        row["injected_histogram"]=hist.tolist() if routes else None
        row["logical_injected_dispatch_bytes_per_rank"]=bytes_dest.tolist() if routes else None
        row["logical_injected_combine_bytes_per_rank"]=bytes_dest.tolist() if routes else None
        row['logical_injected_dispatch_matrix']=traffic.tolist() if routes else None
        rows.append(row)
    return rows


def bootstrap(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    rng=np.random.default_rng(3407)
    samples=values[rng.integers(0,len(values),(5000,len(values)))].mean(axis=1)
    return [float(x) for x in np.percentile(samples,[2.5,97.5])]


def summarize(rows):
    by=defaultdict(list)
    for row in rows:by[(row["arm"],row["modality"])].append(row)
    result=[]
    for (arm,modality),items in by.items():
        summary={"arm":arm,"modality":modality,"n":len(items)}
        for key in ("ttft_ms","lm_start_to_first_token_ms","mean_itl_ms","p95_itl_ms","p99_itl_ms","peak_itl_ms","peak_vsel_ms","slo_violation_fraction","peak_itl_ratio","injection_enqueue_ms"):
            vs=[r[key] for r in items if r[key] is not None]
            summary[key]=None if not vs else {"mean":float(np.mean(vs)),"ci95":bootstrap(vs)}
        result.append(summary)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--runs",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    clean=[];trace=[]
    for run in args.runs:
        if read_json(run/"manifest.json")["trace"]:trace+=trace_rows(run)
        else:clean+=clean_rows(run)
    atomic_json(args.output,{"clean_waves":clean,"clean_summary":summarize(clean),"trace_layers":trace,
        "timing_evidence":"MEASURED_TRUE_EP4_LOCAL_CUDA_EVENT_DURATIONS"})


if __name__=="__main__":main()
