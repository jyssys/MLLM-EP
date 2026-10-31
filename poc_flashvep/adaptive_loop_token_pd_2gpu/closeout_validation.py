"""Check preserved evidence after GPU work. Do not start GPU experiments."""
import collections
import csv
import gzip
import hashlib
import json
import os
import subprocess
import time
from .common import *
from .gate import lines

def main():
    rows=list(csv.DictReader((ROOT/'per_run_metrics.csv').open()))
    clean=[r for r in rows if r['phase'] in ('primary','repeat','protocol_control')]
    phases=collections.Counter(r['phase'] for r in clean)
    assert phases==dict(primary=56,repeat=40,protocol_control=2),phases
    comparisons=0;tokens=collections.Counter();requests=collections.Counter();pd_maps=0
    frozen=json.loads((ROOT/'configs/frozen_execution_hashes.json').read_text())['hashes']
    launches={r['name']:r for r in lines(ROOT/'commands.jsonl')}
    for row in clean:
        name=row['name'];folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text())
        assert row['status']=='COMPLETE' and row['full_depth_work_exact']=='True',name
        values=[r for r in lines(folder/'requests.jsonl') if not r['warmup']]
        assert len(values)==len({r['sample_id'] for r in values})==cfg['count'],name
        assert len(cfg['gpus'])==2 and len(set(cfg['gpus']))==2,name
        launch=launches[name]
        assert launch['quiet'] and launch['gpus']==cfg['gpus'],name
        mask=launch['environment']['CUDA_VISIBLE_DEVICES'].split(',')
        assert mask in ([str(g) for g in cfg['gpus']],[UUIDS[g] for g in cfg['gpus']]),name
        observed={r['physical_GPU']:r['uuid'] for path in folder.glob('NVML-*.jsonl') for r in lines(path)}
        assert set(observed)==set(cfg['gpus']) and all(observed[g]==UUIDS[g] for g in cfg['gpus']),name
        loaded_paths=list(folder.glob('loaded_source_*.json'))
        assert len(loaded_paths)>=3,name
        for loaded in loaded_paths:
            assert json.loads(loaded.read_text())['hashes']==frozen,loaded
        if cfg['arm']!='PD_G1':
            reference=cfg.get('reference_run',f'primary_{cfg["model"]}_{cfg["workload"].lower()}_pd_g1_start0')
            expected={r['sample_id']:r for r in lines(ROOT/'raw'/reference/'requests.jsonl') if not r['warmup']}
            assert len(expected)==cfg['count'],name
            for actual in values:
                assert all(actual[k]==expected[actual['sample_id']][k] for k in ('input_ids','output_ids','exit_depths','finish_reason')),name
                comparisons+=1
        for actual in values:
            assert len(actual['output_ids'])==actual['output_tokens'] and actual['finish_reason']=='length',name
            assert set(actual['exit_depths'])=={cfg['depth']},name
            if cfg['model']=='huginn':assert actual['input_tokens']+actual['output_tokens']<=4096,name
            tokens[cfg['phase']]+=actual['output_tokens'];requests[cfg['phase']]+=1
        mapping=folder/'worker_map.json'
        if mapping.exists():
            physical=json.loads(mapping.read_text());pd_maps+=1
            assert sorted(r['physical_GPU'] for r in physical.values())==cfg['gpus'],name
            assert all(r['uuid'].removeprefix('GPU-')==UUIDS[r['physical_GPU']].removeprefix('GPU-') for r in physical.values()),name
    assert comparisons==1920,comparisons
    assert requests==dict(primary=1568,repeat=640,protocol_control=64),requests
    assert all(r['reconciles']=='True' for r in csv.DictReader((ROOT/'communication_payloads.csv').open()))
    host=collections.defaultdict(float);components={}
    for row in csv.DictReader((ROOT/'e2e_component_shares_aggregate.csv').open()):
        host[row['name']]+=float(row['share_percent'])
        components[row['name'],row['component']]=float(row['share_percent'])
    assert len(host)==98 and all(abs(v-100)<1e-8 for v in host.values()),host
    observations=[]
    for row in clean:
        if row['phase']!='primary' or row['arm']!='Loop_Dynamic':continue
        p,d=json.loads(row['GPU_ids']);name=row['name']
        values={k:components[name,k] for k in ('reservation_queue','P_queue_and_compute_observed','exposed_handoff_observed','generation_observed')}
        observations.append(dict(name=name,model=row['model'],workload=row['workload'],
            mean_P_NVML_utilization_percent=float(row[f'physical_GPU{p}_mean_NVML_utilization_percent']),
            mean_D_NVML_utilization_percent=float(row[f'physical_GPU{d}_mean_NVML_utilization_percent']),
            **values,other_host_intervals=100-sum(values.values()),
            scope='NVML samples plus exclusive host request-time percentages. Not exclusive GPU kernel or DMA shares.'))
    from .analyze import csv_write
    csv_write(ROOT/'closeout_host_and_gpu_observations.csv',observations)
    archive=json.loads((ROOT/'aggregated_request_archive.json').read_text());digestor=hashlib.sha256()
    with gzip.open(ROOT/'per_request.jsonl.gz','rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digestor.update(block)
    assert digestor.hexdigest()==archive['uncompressed_SHA256']
    assert json.loads((ROOT/'state/cpu_test_result.json').read_text())['returncode']==0
    profiles=[r for r in rows if r['phase']=='profile']
    assert len(profiles)==6 and all(r['status']=='COMPLETE' for r in profiles),profiles
    traces=list(csv.DictReader((ROOT/'profiler_rank_summary.csv').open()))
    assert len(traces)==12 and set(collections.Counter(r['name'] for r in traces).values())=={2}
    from .index_guard import main as preserve
    preserve()
    from .resources import base as resources
    owned=resources.burners();active=resources.processes()
    assert set(owned)==set(range(8)),owned
    assert all(os.getsid(r['pid'])==owned[r['gpu']] for r in active),active
    outcome=dict(passed=True,time=time.time(),elapsed_hours=(time.time()-START_EPOCH)/3600,
        clean_server_starts=dict(phases),clean_measured_requests=dict(requests),output_tokens=dict(tokens),
        exact_nonreference_output_comparisons=comparisons,measured_PD_worker_UUID_maps=pd_maps,
        DP_mapping_scope='DP masks and observed physical-GPU samples. DP does not emit the PD worker UUID-map artifact.',
        frozen_source_hashes_passed=True,archive_SHA256_passed=True,host_share_sum_passed=True,
        profiler_jobs=6,profiler_rank_traces=12,owned_burners=owned,GPU_processes=active,
        scope='CPU verification of measured artifacts and returned resources. No new GPU serving experiment.')
    write(ROOT/'state/final_validation.json',outcome)
    budget(status='COMPLETE_REVIEWED',final_validation_passed=True,active=[],current_gpu_ownership=[])
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
