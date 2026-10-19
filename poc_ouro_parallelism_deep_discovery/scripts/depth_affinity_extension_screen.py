"""Repeat the declared unit-cost ownership screen on admitted new traces."""
import argparse
import datetime as dt
import json
from pathlib import Path
import random

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha
from poc_ouro_parallelism_deep_discovery.scripts.depth_affinity_capacity_screen import screen,visits
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg


def convert(row):
    return dict(N_decode_events=row['recorded_decode_positions'],
        R2_events=row['R2_positions'],R3_events=row['R3_positions'],R4_events=row['R4_positions'],
        executed_depth_steps=row['recorded_decode_core_pass_work'])


def analyze(source):
    value=read(source)
    if value['evidence_class']!='QUALITY_ONLY':
        raise ValueError('Use admitted natural baseline traces')
    available={(r['size'],r['mode']) for r in value['jobs'] if r['status']=='AVAILABLE'}
    rng=random.Random(202610191330)
    records=[]
    for size in ('1.4B','2.6B'):
        if (size,'adaptive') not in available:
            continue
        for domain in ('gsm8k','math500'):
            original=[r for r in value['records'] if (r['size'],r['mode'],r['dataset'])==(size,'adaptive',domain)]
            if len(original)!=32 or len({r['id'] for r in original})!=32:
                raise ValueError('Require a complete unique32-request domain')
            rows=[convert(r) for r in original if r['recorded_decode_positions']]
            if not rows:
                raise ValueError('No decode work for ownership screen')
            for P in (2,4,8):
                result=screen(visits(rows),P)
                bootstrap=sorted(screen(visits(rng.choices(rows,k=len(rows))),P)[
                    'pinned_over_pooled_saturated_capacity'] for _ in range(4000))
                result.update(size=size,dataset=domain,N_requests=len(original),
                    N_requests_with_decode=len(rows),request_bootstrap_95_CI=[bootstrap[100],bootstrap[3899]],
                    hardware_status='P8_UNMEASURED_SCENARIO' if P==8 else 'CPU_UNIT_COST_MODEL')
                records.append(result)
    return dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),records=records,
        fixed_R4=[screen([1,1,1,1],P) for P in (2,4,8)],
        input_sha256={str(source.relative_to(ROOT)):sha(source),
            'scripts/depth_affinity_capacity_screen.py':sha(ROOT/'scripts/depth_affinity_capacity_screen.py')},
        evidence_class='ANALYTIC_BOUND',quality='No new candidate generation',
        meaning='Service-capacity expression inside the inherited unit-cost, zero-transfer model. Not measured latency or GPU throughput.',
        assumptions=['Each complete core task has the same unit cost at every depth.',
            'Every worker retains the complete shared weights.',
            'Each logical depth has one fixed compute/KV owner.',
            'Transfers,gate,head,state publication and contention cost zero.',
            'The comparison pool is ideal and work-conserving,not an implemented CDB runtime.',
            'Search every permitted static depth partition using the observed histogram. This is hindsight,not a causal policy.'],
        limitations=['No physical latency lower bound or same-P speedup follows.',
            'Batch efficiency,KV ownership,real transfers and different granularity can change results.',
            'Do not reject every depth-dependent physical placement from this one layout.',
            'No P8 measurement or topology assumption enters the unit-cost arithmetic.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value=analyze(args.source.resolve())
    args.output.mkdir(parents=True)
    (args.output/'screen.json').write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    for size in ('1.4B','2.6B'):
        rows=[r for r in value['records'] if r['size']==size and r['P'] in (2,4)]
        if rows:
            series=[dict(name=f"{r['dataset']} P{r['P']}",points=[(x['C'],x['pinned_over_pooled_capacity'])
                for x in r['finite_C']]) for r in rows]
            svg(args.output/f'{size}_unit_cost_capacity.svg',series,
                f'Base{size}: declared unit-cost ownership screen','Independent requests C','Modeled pinned/pool capacity ratio')
    print(json.dumps(dict(structural_cells=len(value['records']),evidence_class=value['evidence_class'])))


if __name__=='__main__':
    main()
