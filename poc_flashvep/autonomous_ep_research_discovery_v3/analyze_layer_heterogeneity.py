#!/usr/bin/env python3
"""Report layer heterogeneity and its mass contribution."""
from __future__ import annotations
import argparse,csv,statistics
from collections import defaultdict
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--atlas',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 by=defaultdict(list)
 for r in csv.DictReader(open(a.atlas)):
  by[(r['phase'],int(float(r['layer'])))].append(float(r['cuda_ms']))
 rows=[]
 for (ph,l),v in sorted(by.items()):
  rows.append({'phase':ph,'layer':l,'n':len(v),'p50_ms':statistics.median(v),'p90_ms':sorted(v)[int(.9*(len(v)-1))]})
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Layer heterogeneity','', 'Fresh logical rows grouped by phase/layer. This is a descriptive control; layer 0/early differences are not promoted unless they carry material request-level mass.', '', '|phase|layer|n|T_MoE p50|p90|','|---|---:|---:|---:|---:|']
 for r in rows: lines.append(f"|{r['phase']}|{r['layer']}|{r['n']}|{r['p50_ms']:.3f}|{r['p90_ms']:.3f}|")
 out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
 print({'rows':len(rows),'out':str(out)})
if __name__=='__main__':main()
