#!/usr/bin/env python3
"""Aggregate the bounded ASAP reproduction outputs and generate figures."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(root: Path) -> pd.DataFrame:
    rows=[]
    for p in sorted((root/"asap_raw").glob("asap_rank*.jsonl")):
        rank=int(p.stem.split("rank")[-1])
        for line in p.read_text().splitlines():
            if line:
                x=json.loads(line); x["ep_rank_file"]=rank; rows.append(x)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--result", type=Path, nargs="+", required=True); ap.add_argument("--output", type=Path, required=True); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    frames=[]
    for root in a.result:
        d=load(root); d.to_csv(root/"asap_event_rows.csv",index=False); frames.append(d)
    df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(); df.to_csv(a.output/"asap_event_rows_all.csv",index=False)
    summary=[]
    if not df.empty:
        for (root,mode,scale,delay,measured),g in df.groupby(["ep_rank_file","mode","scale","delay_ms","measured"]):
            summary.append({"rank":root,"mode":mode,"scale":scale,"delay_ms":delay,"measured":bool(measured),"n":len(g),"pre_moe_median_ms":g.pre_moe_cuda_ms.median(),"ep_done_median_ms":g.ep_entry_to_done_ms.median(),"event_wait_median_ms":g.event_wait_cuda_ms.median(),"event_wait_p95_ms":g.event_wait_cuda_ms.quantile(.95),"injected_delay_median_ms":g.get("injected_delay_cuda_ms",pd.Series(dtype=float)).median()})
    pd.DataFrame(summary).to_csv(a.output/"asap_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    if not df.empty:
        q=df[df.measured==True].groupby(["delay_ms","ep_rank_file"]).pre_moe_cuda_ms.median().unstack()
        q.plot(ax=ax,marker="o")
    ax.set(xlabel="injected GPU delay (ms)",ylabel="pre-MoE CUDA duration (ms)",title="Positive-control response"); fig.tight_layout(); fig.savefig(a.output/"plot1_injected_skew_response.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    if not df.empty:
        q=df[df.measured==True].groupby(["delay_ms","ep_rank_file"]).event_wait_cuda_ms.median().unstack(); q.plot(ax=ax,marker="o")
    ax.set(xlabel="injected GPU delay (ms)",ylabel="DeepEP EventOverlap wait (ms)",title="Direct event-wait response"); fig.tight_layout(); fig.savefig(a.output/"plot2_event_wait_response.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if not df.empty:
        q=df[df.measured==True].groupby(["mode","scale"]).ep_entry_to_done_ms.median(); q.plot(ax=ax,kind="bar")
    ax.set_ylabel("EP entry→done CUDA ms"); fig.tight_layout(); fig.savefig(a.output/"plot3_balanced_heterogeneous.png",dpi=160); plt.close(fig)
    print(a.output)
if __name__=="__main__": main()
