"""Device rotation and process-lifetime telemetry, not critical-path profiling."""
import json
import numpy as np
from .common import ROOT
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    rotation=[];telemetry=[];cohorts=[]
    for folder in sorted((ROOT/'raw').glob('*')):
        identity=folder/'identity.json'
        if not identity.exists():continue
        data=json.loads(identity.read_text());job=data['job'];done=(folder/'completed.json').exists()
        if done and job['kind']=='dual_serving':
            for result in folder.glob('*.json'):
                record=json.loads(result.read_text())
                if isinstance(record,dict) and 'scenario' in record and 'label' in record:
                    cohorts.append(dict(run_id=folder.name,policy=record['label'],scenario=record['scenario']['id'],
                        lane=record['lane'],GPU=data['physical_GPU'],start=record['start'],end=record['end']))
        for i,cell in enumerate(job.get('cells',[])):
            rotation.append(dict(run_id=folder.name,GPU=data['physical_GPU'],UUID=data['UUID'],pid=data['pid'],
                completed=done,policy_order=i,policy=cell['label'],runtime=cell['runtime'],C_cap=cell['C'],
                surface=cell.get('surface','NATIVE'),fairness=cell.get('fairness','completion'),
                B_ms=cell.get('budget_ms'),width=job.get('width'),cap=job.get('cap'),lane=job.get('lane',job['kind']),
                scope='One model process per authorized GPU. Every listed cell executes on that same GPU; policy order rotates.'))
        file=folder/'telemetry.jsonl'
        if not file.exists():continue
        samples=[]
        for line in file.read_text().splitlines():
            try:
                row=json.loads(line);parts=[x.strip() for x in row['stats'].split(',')]
                assert parts[0]==data['UUID']
                if len(parts)<6:continue
                values=[float(x) for x in parts[1:6]]
                samples.append(values)
            except (KeyError,ValueError):continue
        if not samples:continue
        values=np.array(samples)
        out=dict(run_id=folder.name,GPU=data['physical_GPU'],completed=done,samples=len(values),
            scope='Whole process lifetime, including startup, warm-up, drain and between-cell work. Not GPU busy or critical-path attribution.')
        for col,name in enumerate(('GPU_utilization_pct','GPU_memory_MiB','GPU_power_W','SM_clock_MHz','temperature_C')):
            out[name+'_P50']=float(np.median(values[:,col]));out[name+'_min']=float(values[:,col].min());out[name+'_max']=float(values[:,col].max())
        telemetry.append(out)
    csv_file(ROOT/'GPU_policy_rotation.csv',rotation);csv_file(ROOT/'GPU_process_telemetry.csv',telemetry)
    overlaps=[]
    for row in cohorts:
        others=[r for r in cohorts if r['GPU']!=row['GPU'] and r['start']<row['end'] and r['end']>row['start']]
        seconds=sum(max(0,min(row['end'],r['end'])-max(row['start'],r['start'])) for r in others)
        overlaps.append(dict(row,mean_other_measurement_cohorts_active=seconds/(row['end']-row['start']),
            other_measurement_GPUs=','.join(str(x) for x in sorted({r['GPU'] for r in others})),
            scope='Shared-host perf_counter intervals across recorded serving windows. Excludes other workers startup/warmup and all burner activity.'))
    csv_file(ROOT/'cohort_coexecution.csv',overlaps)
    print('Resource records',len(rotation),len(telemetry))

if __name__=='__main__':main()
