"""Normalize R1/R4 hardware controls without treating R1 as a quality baseline."""
import argparse
import csv
import datetime as dt
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg, table


def load(label):
    with (ROOT/'analysis'/label/'native_summary.csv').open() as stream:
        return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    args=p.parse_args()
    output=ROOT/'analysis'/args.label
    output.mkdir(parents=True,exist_ok=False)
    controls={}
    for R,label in [(1,'async_R1_checkpoint_01'),(4,'async_checkpoint_01')]:
        for row in load(label):
            key=(row['model'],R,int(row['P']),int(row['context']))
            assert key not in controls
            controls[key]=row
    records=[]
    for size in MODELS:
        for N in [1024,4096,8192]:
            for P in [1,2,4]:
                one,four=[controls[size,R,P,N] for R in [1,4]]
                item=dict(model=size,P=P,context=N,evidence_class='LIVE_E2E',
                          workload='128 forced output tokens, two frozen prompt shapes, ten trials per condition.',
                          quality='R1 is a hardware ablation. It is not quality-equivalent to R4.')
                for metric in ['wall','TTFT','TPOT']:
                    field=f'median_{metric}_ms'
                    a,b=float(one[field]),float(four[field])
                    speed1=float(controls[size,1,1,N][field])/a
                    speed4=float(controls[size,4,1,N][field])/b
                    item.update({f'R1_{metric}_ms':a,f'R4_{metric}_ms':b,
                                 f'R4_over_R1_{metric}':b/a,
                                 f'R1_{metric}_speedup_vs_TP1':speed1,
                                 f'R4_{metric}_speedup_vs_TP1':speed4,
                                 f'R4_over_R1_{metric}_TP_scaling_ratio':speed4/speed1})
                records.append(item)
    table(output/'recurrence_controls.csv',records)
    for size in MODELS:
        for metric in ['wall','TPOT']:
            series=[]
            for R in [1,4]:
                for N in [1024,8192]:
                    series.append(dict(name=f'R{R}, N{N}',points=[
                        (row['P'],row[f'R{R}_{metric}_speedup_vs_TP1'])
                        for row in records if row['model']==size and row['context']==N]))
            svg(output/f'{size}_{metric}_scaling.svg',series,
                f'{size}: R1 hardware ablation versus R4','Physical TP degree',f'{metric} speedup versus same-R TP1 (x)')
    write(output/'summary.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),records=records,
          interpretation='Ratios describe measured controlled workloads. They do not isolate recurrence from changed graph size, effective depth, and depth-indexed KV.',
          limitations=['Two request shapes. Sequential configuration order. No broad request-level confirmation.',
                       'Native R1 and R4 have different graph structure and executed depth.',
                       'No new candidate architecture, adaptive result, or quality-equivalence claim.']))
    for row in records:
        if row['P']==4:
            print(row['model'],row['context'],'TPOT scaling R1/R4',
                  round(row['R1_TPOT_speedup_vs_TP1'],4),round(row['R4_TPOT_speedup_vs_TP1'],4))


if __name__=='__main__':main()
