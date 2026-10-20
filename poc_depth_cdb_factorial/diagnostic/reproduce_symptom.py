"""Replay the archived C64 summary. Exit1 if hindsight misses the stated gain."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('poc_depth_cdb_factorial/artifacts/metrics_raw.csv'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expect-gain-at-least',type=float,default=1.05)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=list(csv.DictReader(args.source.open()))
    selected={arm:[r for r in rows if r['kind']=='speed' and r['C']=='64' and
              r['arm']==arm and r['trace']=='False'] for arm in ('A0','A1','A2')}
    assert all(len(r)==5 and len({x['repeat'] for x in r})==5 for r in selected.values())
    medians={arm:statistics.median(float(r['makespan_s']) for r in rows) for arm,rows in selected.items()}
    ratio=medians['A1']/medians['A2']
    result=dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        evidence_class='LIVE_MEASURED',reanalysis='Archived live measurements, not a new GPU run.',
        C=64,repeats=5,median_makespan_s=medians,A1_over_A2=ratio,
        expected_minimum_gain=args.expect_gain_at_least,
        status='GAIN_CONFIRMED' if ratio>=args.expect_gain_at_least else 'SYMPTOM_REPRODUCED',
        meaning='A red result reproduces the missing >=5% gain. It does not identify the cause or authorize a fix.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))
    raise SystemExit(int(ratio<args.expect_gain_at_least))


if __name__=='__main__':main()
