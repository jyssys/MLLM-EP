"""Portable raw outputs and actual plans, without large source/telemetry copies."""
import gzip
import json
from contextlib import ExitStack
from .common import ROOT,digest,write

def main():
    assert json.loads((ROOT/'evidence_validation.json').read_text())['all_complete']
    names=['portable_serving_outputs.jsonl.gz','portable_quality_outputs.jsonl.gz',
        'portable_actual_prefill_plans.jsonl.gz','portable_online_decisions.jsonl.gz','portable_event_service.jsonl.gz']
    counts={name:0 for name in names}
    with ExitStack() as stack:
        streams={name:stack.enter_context(gzip.open(ROOT/name,'wt',compresslevel=6)) for name in names}
        def emit(name,value):
            streams[name].write(json.dumps(value,separators=(',',':'))+'\n');counts[name]+=1
        for folder in sorted((ROOT/'raw').glob('*')):
            if not (folder/'completed.json').exists():continue
            for file in sorted(folder.glob('*.json')):
                if file.name in ('dual_sources.json','harness_sources.json','completed.json','identity.json'):continue
                data=json.loads(file.read_text())
                if not isinstance(data,dict):continue
                common=dict(source=str(file.relative_to(ROOT)),source_sha256=digest(file),run_id=folder.name)
                if 'scenario' in data and 'label' in data:
                    base=dict(common,policy=data['label'],cell=data['cell'],scenario=data['scenario'],lane=data['lane'],
                        physical_GPU=data['physical_GPU'],full_R4=data['full_R4'])
                    emit(names[0],dict(base,requests=data['requests'],input_families=data['families'],
                        start=data['start'],end=data['end'],queue_series=data['queue_series'],
                        configured_N=data['configured_N'],arrival_window_s=data['arrival_window_s']))
                    emit(names[2],dict(base,dispatch_descriptors=data['dispatch_descriptors'],
                        prompt_token_loop_work=data['prompt_token_loop_work'],chunk_decisions=data['chunk_decisions']))
                    if data['optimizer_decisions']:
                        emit(names[3],dict(base,optimizer_decisions=data['optimizer_decisions'],
                            admission_snapshots=data['admission_snapshots'],snapshot_calls=data['snapshot_calls'],
                            overbudget_admission=data['overbudget_admission']))
                    if data['service_observations']:emit(names[4],dict(base,service_observations=data['service_observations']))
                elif folder.name.startswith('dual_p4_QUALITY_') and 'output_ids' in data and 'id' in data:
                    emit(names[1],dict(common,record=data))
    artifacts=[]
    for name in names:
        file=ROOT/name;assert file.stat().st_size<90*2**20,('Split this portable artifact before Git',name)
        artifacts.append(dict(path=name,rows=counts[name],bytes=file.stat().st_size,sha256=digest(file)))
    write(ROOT/'portable_evidence_manifest.json',dict(artifacts=artifacts,
        scope='Lossless selected raw fields from completed attempts. Failed attempts, full source snapshots, logs, telemetry, and other diagnostics remain in the local raw archive.'))
    print(json.dumps(artifacts,indent=2))

if __name__=='__main__':main()
