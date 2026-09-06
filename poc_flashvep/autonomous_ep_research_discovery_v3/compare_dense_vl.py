#!/usr/bin/env python3
"""Emit a compact cross-model control; models are never pooled."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def read(p):
    z=json.loads(Path(p).read_text())
    out={"e2e": z.get("e2e"), "stage": {}}
    for ph,s in z.get("stage_summary",{}).items():
        out["stage"][ph]={"n":s.get("n"), "tmoe_p50_ms":s.get("cuda_p50_ms"),
            "tmoe_p99_ms":s.get("cuda_p99_ms"),
            "dispatch_p50_ms":s.get("deepep_dispatch",{}).get("p50_ms"),
            "expert_p50_ms":s.get("expert",{}).get("p50_ms"),
            "event_wait_p50_ms":s.get("deepep_event_wait",{}).get("p50_ms")}
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--vl",required=True); ap.add_argument("--dense",nargs=2,required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    z={"protocol":"fixed text, TP2/DP2/EP4, DeepEP HT, BF16, eager; models kept separate",
       "qwen3_vl":read(a.vl), "qwen3_dense_rep1":read(a.dense[0]), "qwen3_dense_rep2":read(a.dense[1])}
    Path(a.out).write_text(json.dumps(z,indent=2),encoding="utf-8")
if __name__=="__main__": main()
