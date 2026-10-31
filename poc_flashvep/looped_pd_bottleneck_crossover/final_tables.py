"""Summarize observed transport counters and finite rate-grid outcomes."""
import collections
import csv
from .common import *
from .summary_tables import records,save_csv


def transport():
    rows=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        path=ROOT/'configs'/(folder.name+'.json')
        if not path.exists():continue
        cfg=json.loads(path.read_text())
        before=folder/'before_measured.json';after=folder/'after_measured.json'
        if before.exists() and after.exists():
            a=json.loads(before.read_text());b=json.loads(after.read_text())
            if not all(isinstance(v,dict) for v in b.values()):continue
            for worker,observed in b.items():
                base=a.get(worker,{})
                row=dict(name=folder.name,model=cfg['model'],worker=worker,arm=cfg.get('arm'),
                    phase=cfg.get('phase'),cell=cfg.get('workload'),scope='Measured-window counter delta excludes warm-up.',
                    evidence='MEASURED_RUNTIME_COUNTERS',pure_GPU_DMA_time_s='NOT_MEASURED',
                    overlap_fraction='NOT_MEASURED_EXCLUSIVE_CRITICAL_PATH')
                for field in ('transport_bytes_sent','transport_completed_count','transport_completion_observation_seconds',
                    'issued_loop_rows','issued_submissions'):
                    if field in observed and field in base:row[field]=observed[field]-base[field]
                row['duration_scope']='Host-observed cumulative transfer completion time. Transfers can overlap. Not an E2E component.'
                rows.append(row)
        if cfg['model']!='qwen':continue
        for role in ('prefill','decode'):
            path=folder/f'ar_native_{role}.json'
            if not path.exists():continue
            data=json.loads(path.read_text());intervals=data.get('NIXL_stats_intervals',[])
            flat=collections.defaultdict(list)
            for interval in intervals:
                for key,values in interval['data'].items():flat[key].extend(values)
            done=data.get('NIXL_done_events',[])
            row=dict(name=folder.name,model='qwen',worker=role,phase=cfg.get('phase'),cell=cfg.get('workload'),
                evidence='MEASURED_NATIVE_CONNECTOR_TELEMETRY',scope='Process lifetime includes warm-up.',
                done_events=len(done),done_request_ids=sorted({rid for event in done for rid in event['request_ids']}),
                telemetry_keys=sorted(flat),pure_GPU_DMA_time_s='NOT_MEASURED',
                overlap_fraction='NOT_MEASURED_EXCLUSIVE_CRITICAL_PATH')
            for key,values in flat.items():
                if values and all(isinstance(v,(int,float)) for v in values):
                    row[key+'_sum']=sum(values);row[key+'_observations']=len(values)
            row['counter_status']='AVAILABLE' if flat else 'NOT_AVAILABLE_EMPTY_NATIVE_COUNTERS'
            row['duration_scope']='Returned native telemetry. No final/interval double counting. DONE snapshots are not summed.'
            rows.append(row)
    save_csv(ROOT/'transport_runtime_counters.csv',rows)


def rate_grid():
    rows=list(csv.DictReader((ROOT/'serving_runs.csv').open()))
    groups=collections.defaultdict(list)
    for row in rows:
        if row['eligible_for_serving_comparison']!='True':continue
        cfg=json.loads((ROOT/'configs'/(row['name']+'.json')).read_text())
        if 'rate_factor' not in cfg:continue
        key=tuple(row.get(k) for k in ('model','cell','arm','chunk','KV_blocks','compute_slots','offered_requests'))
        groups[key].append(row)
    output=[]
    for key,values in groups.items():
        passed=[r for r in values if float(r['joint_SLO_attainment'])>=.9 and int(r['failed_or_missing'])==0]
        row=dict(zip(('model','cell','arm','chunk','KV_blocks','compute_slots','requests_per_start'),key))
        row.update(observed_starts=len(values),distinct_target_rates=len({r['arrival_rate'] for r in values}),
            starts_with_at_least_90_percent_attainment=len(passed),
            highest_passing_target_rate=max((float(r['arrival_rate']) for r in passed),default='NO_PASSING_POINT'),
            maximum_observed_finite_goodput=max(float(r['SLO_goodput_requests_per_s']) for r in values),
            all_starts_90_percent_at_highest_passing_rate='NOT_APPLICABLE',
            stability='NOT_ESTABLISHED_FINITE_ARRIVAL_AND_DRAIN',
            evidence='TRACE_ANALYSIS_OF_MEASURED_OPEN_LOOP_RUNS',
            scope='Maximum passing point on the measured finite grid. Not a sustainable-capacity estimate.')
        if passed:
            top=row['highest_passing_target_rate'];same=[r for r in values if float(r['arrival_rate'])==top]
            row['all_starts_90_percent_at_highest_passing_rate']=all(float(r['joint_SLO_attainment'])>=.9 for r in same)
            row['starts_at_highest_passing_rate']=len(same)
        output.append(row)
    save_csv(ROOT/'finite_rate_grid_summary.csv',output)


def screen_tables():
    rows=list(csv.DictReader((ROOT/'serving_runs.csv').open()))
    clean=[r for r in rows if r['eligible_for_serving_comparison']=='True']
    screen=[r for r in clean if '_screen_start0' in r['name']]
    controls=[r for r in clean if '_dp_' in r['name'] or '_dp_start' in r['name'] or '_tuned_chunk_' in r['name']]
    save_csv(ROOT/'primary_screen_comparison.csv',screen)
    save_csv(ROOT/'generic_runtime_controls.csv',controls)
    write(ROOT/'state/clean_measurement_totals.json',dict(time=time.time(),
        clean_runs=len(clean),clean_offered_requests=sum(int(r['offered_requests']) for r in clean),
        clean_completed_requests=sum(int(r['completed_requests']) for r in clean),
        by_model={m:dict(runs=sum(r['model']==m for r in clean),
            requests=sum(int(r['offered_requests']) for r in clean if r['model']==m)) for m in ('ouro','huginn','qwen')},
        scope='Exclude calibration, heavy qualification, profiling, and prepared-Decode diagnostics.'))


def main():
    transport();rate_grid();screen_tables();print('FINAL_TABLES_COMPLETE')


if __name__=='__main__':main()
