#!/usr/bin/env python3
"""Summarize shape-transition controls without mixing them into the atlas."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else None


def transition(root: Path) -> dict:
    rows = []
    for line in (root / "invocations.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("active_wave") is None or int(r.get("layer", -1)) < 0:
            continue
        if int(r.get("M", 0) or 0) > 2048:
            continue
        # Collapse TP/EP rows within one logical DP invocation; do not sum
        # ranks because they are correlated views of the same critical span.
        rows.append(r)
    groups = {}
    for r in rows:
        k = (int(r.get("active_wave")), int(r.get("dp_rank", -1)),
             int(r.get("local_invocation_id", -1)), int(r.get("layer", -1)),
             str(r.get("phase", "unknown")))
        z = groups.setdefault(k, {"wave": k[0], "phase": k[4], "cuda": [], "wait": []})
        z["cuda"].append(float(r.get("cuda_ms", 0) or 0))
        for s in r.get("stage_records", []) or []:
            if s.get("stage") == "deepep_event_wait":
                z["wait"].append(float(s.get("cuda_ms", 0) or 0))
    waves = {}
    for z in groups.values():
        w = waves.setdefault(z["wave"], {"cuda": [], "wait": [], "phases": {}})
        c = max(z["cuda"]) if z["cuda"] else 0.0
        q = max(z["wait"]) if z["wait"] else 0.0
        w["cuda"].append(c); w["wait"].append(q)
        ph = w["phases"].setdefault(z["phase"], {"cuda": [], "wait": []})
        ph["cuda"].append(c); ph["wait"].append(q)
    out = []
    for wave in sorted(waves):
        w = waves[wave]
        out.append({"wave": wave, "n": len(w["cuda"]),
                    "tmoe_p50_ms": med(w["cuda"]), "wait_p50_ms": med(w["wait"]),
                    "phase": {p: {"tmoe_p50_ms": med(v["cuda"]),
                                   "wait_p50_ms": med(v["wait"]), "n": len(v["cuda"])}
                              for p, v in sorted(w["phases"].items())}})
    e2e = []
    for p in sorted(root.glob("waves.dp*.json")):
        try: e2e.extend(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError): pass
    e2e_by_wave = {}
    for w in e2e:
        e2e_by_wave.setdefault(int(w.get("wave", -1)), []).append(float(w.get("wall_ms", 0) or 0))
    for x in out: x["wall_p50_ms"] = med(e2e_by_wave.get(x["wave"], []))
    return {"trace": root.name, "waves": out,
            "n_logical": sum(x["n"] for x in out),
            "e2e_waves": len(e2e_by_wave)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    z = [transition(p) for p in a.root]
    a.out.write_text(json.dumps(z, indent=2), encoding="utf-8")
    print(json.dumps({"traces": len(z), "out": str(a.out)}, indent=2))


if __name__ == "__main__": main()
