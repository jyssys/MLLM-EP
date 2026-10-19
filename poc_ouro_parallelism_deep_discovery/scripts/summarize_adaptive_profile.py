"""Describe original-gate calibration traces. No scheduling or quality gain claim."""
import argparse
import collections
import datetime as dt
import random
import statistics
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg,table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True)
    args=parser.parse_args();output=ROOT/'analysis'/args.label
    output.mkdir(parents=True,exist_ok=False)
    requests=[]
    for root in sorted((ROOT/'outputs').glob('quality_base*_adaptive_v1')):
        for directory in root.iterdir():
            if not directory.is_dir() or not (directory/'status.json').exists():continue
            assert read(directory/'status.json').get('success')
            for path in sorted(directory.glob('*.json')):
                row=read(path)
                if 'depth_events' not in row:continue
                events=row['depth_events'];depths=[v['depth'] for v in events]
                assert all(v in [2,3,4] for v in depths)
                assert len(depths)==row['generated_tokens']-1
                assert sum(depths)==row['stats']['recurrent_steps']
                assert row['stats']['fixed_depth_recurrent_steps']==4*len(depths)
                histogram=collections.Counter(depths)
                record=dict(model=row['size'],id=row['id'],dataset=row['dataset'],N_decode_events=len(depths),
                            generated_tokens=row['generated_tokens'],executed_depth_steps=sum(depths),
                            counterfactual_same_length_fixed_steps=4*len(depths),mean_depth=statistics.mean(depths),
                            R2_events=histogram[2],R3_events=histogram[3],R4_events=histogram[4],
                            loop_work_reduction=1-sum(depths)/(4*len(depths)),
                            prefill_depth_steps=row['stats']['prefill_recurrent_steps'],
                            natural_wall_ms=row['wall_ms'],finish_reason=row['finish_reason'],
                            source=str(path.relative_to(ROOT)))
                requests.append(record)
    table(output/'request_profiles.csv',requests)
    summaries=[];rng=random.Random(202610191243)
    for size in MODELS:
        for domain in ['gsm8k','math500']:
            rows=[r for r in requests if r['model']==size and r['dataset']==domain]
            assert len(rows)==8
            total=sum(r['N_decode_events'] for r in rows)
            work=sum(r['executed_depth_steps'] for r in rows)
            bootstrap=[]
            for _ in range(4000):
                chosen=rng.choices(rows,k=len(rows))
                bootstrap.append(1-sum(r['executed_depth_steps'] for r in chosen)/(4*sum(r['N_decode_events'] for r in chosen)))
            bootstrap.sort()
            summaries.append(dict(model=size,dataset=domain,N_requests=len(rows),N_decode_events=total,
                                  pooled_mean_depth=work/total,
                                  mean_request_depth=statistics.mean(r['mean_depth'] for r in rows),
                                  fraction_R2=sum(r['R2_events'] for r in rows)/total,
                                  fraction_R3=sum(r['R3_events'] for r in rows)/total,
                                  fraction_R4=sum(r['R4_events'] for r in rows)/total,
                                  counterfactual_same_length_loop_work_reduction=1-work/(4*total),
                                  request_bootstrap_95_CI=[bootstrap[100],bootstrap[3899]]))
        ordered=sorted([r for r in requests if r['model']==size],key=lambda r:(r['dataset'],r['id']))
        svg(output/f'{size}_mean_depth.svg',
            [dict(name='Original online gate',points=[(i+1,r['mean_depth']) for i,r in enumerate(ordered)])],
            f'{size}: frozen original-gate calibration','Request index','Mean executed decode depth')
    write(output/'summary.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),summaries=summaries,
          evidence_class='QUALITY_ONLY',sample_unit='Request. Tokens are not independent bootstrap samples.',
          protocol='Original checkpoint gate, q0.5, minimumR2, maximumR4, last_exited, fullR4 prefill, greedy natural output.',
          interpretation='Work counts compare the observed adaptive output length with four steps per same-length decode event. They are not measured speedups or quality-equivalent fixed-mode comparisons.',
          candidate='NONE',held_out='UNTOUCHED',HBM_traffic='NOT_MEASURED'))
    for row in summaries:
        print(row['model'],row['dataset'],'mean_depth',round(row['pooled_mean_depth'],4),
              'same_length_work_reduction',round(row['counterfactual_same_length_loop_work_reduction'],4))


if __name__=='__main__':main()
