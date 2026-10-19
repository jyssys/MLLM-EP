"""Describe completed online depth traces. No scheduling or prediction policy."""
import argparse
import collections
import csv
import datetime as dt
import gzip
import json
import math
from pathlib import Path
import statistics
from poc_ouro_parallelism_deep_discovery.contract import ROOT,read,sha
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg


def distribution(depths):
    if not depths or any(type(d) is not int or d not in (2,3,4) for d in depths):
        raise ValueError('Require nonempty observed depths2/3/4')
    counts=collections.Counter(depths)
    return [counts[d]/len(depths) for d in (2,3,4)]


def jsd(left,right):
    if len(left)!=len(right) or not left or any(x<0 for x in left+right):
        raise ValueError('Require equal nonnegative distributions')
    if not math.isclose(sum(left),1) or not math.isclose(sum(right),1):
        raise ValueError('Distributions must sum to1')
    middle=[(a+b)/2 for a,b in zip(left,right)]
    return sum(0.5*p*math.log(p/m) for side in (left,right) for p,m in zip(side,middle) if p)


def quarters(depths):
    if len(depths)<4:
        raise ValueError('Require four nonempty descriptive quarters')
    return [depths[len(depths)*q//4:len(depths)*(q+1)//4] for q in range(4)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source=ROOT/'analysis/baseline_extension_complete_v1/summary.json'
    admitted=read(source)
    hashes={str(source.relative_to(ROOT)):sha(source)}
    requests=[]
    for row in admitted['records']:
        if row['mode']!='adaptive':
            continue
        path=ROOT/row['source']
        if sha(path)!=admitted['input_sha256'][row['source']]:
            raise ValueError('Admitted depth record changed')
        hashes[row['source']]=sha(path)
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            raw=json.load(stream)
        depths=[event['depth'] for event in raw['depth_events']]
        if len(depths)!=row['recorded_decode_positions'] or sum(depths)!=row['recorded_decode_core_pass_work']:
            raise ValueError('Raw depths disagree with admitted work')
        parts=quarters(depths)
        probabilities=distribution(depths)
        requests.append(dict(size=row['size'],dataset=row['dataset'],id=row['id'],positions=len(depths),
            R2=depths.count(2),R3=depths.count(3),R4=depths.count(4),mean_depth=statistics.mean(depths),
            R2_fraction=probabilities[0],R3_fraction=probabilities[1],R4_fraction=probabilities[2],
            quarter_means=[statistics.mean(part) for part in parts],
            early_late_JSD_nats=jsd(distribution(parts[0]),distribution(parts[-1]))))
    cells=[]
    for size,domain in sorted({(r['size'],r['dataset']) for r in requests}):
        rows=[r for r in requests if (r['size'],r['dataset'])==(size,domain)]
        if len(rows)!=32 or len({r['id'] for r in rows})!=32:
            raise ValueError('Require complete unique32-request cells')
        total=sum(r['positions'] for r in rows)
        cells.append(dict(size=size,dataset=domain,N_requests=len(rows),recorded_positions=total,
            token_weighted_depth_distribution=[sum(r[f'R{d}'] for r in rows)/total for d in (2,3,4)],
            request_weighted_depth_distribution=[statistics.mean(r[f'R{d}_fraction'] for r in rows) for d in (2,3,4)],
            mean_early_late_JSD_nats=statistics.mean(r['early_late_JSD_nats'] for r in rows),
            request_bootstrap_JSD_CI=interval([r['early_late_JSD_nats'] for r in rows]),
            normalized_position_quarters=[dict(quarter=q+1,mean_depth=statistics.mean(r['quarter_means'][q] for r in rows),
                request_bootstrap_CI=interval([r['quarter_means'][q] for r in rows])) for q in range(4)]))
    result=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),requests=requests,cells=cells,input_sha256=hashes,
        evidence_class='QUALITY_ONLY',scope='Descriptive original-gate trace characterization. No candidate, predictor, scheduling or speed result.',
        denominators='Recorded decode positions exclude fullR4prefill and its first generated token. Quarters use relative position within each completed trace.',
        uncertainty='4000 request-resampling bootstrap,seed2026101911. Tokens and quarters are not independent requests.',
        causal_limit='Both the late trace and final length are future information. Descriptive quarter statistics are not online scheduling inputs.')
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (args.output/'requests.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(requests[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(requests)
    svg(args.output/'depth_by_relative_position.svg',[
        dict(name=f"{r['size']} {r['dataset']}",points=[(q['quarter'],q['mean_depth']) for q in r['normalized_position_quarters']])
        for r in cells],'Original-gate traces: descriptive relative-position quarters','Trace quarter,using future final length','Request-mean executed decode depth')
    print(json.dumps(dict(requests=len(requests),cells=len(cells),candidate='NONE')))


if __name__=='__main__':
    main()
