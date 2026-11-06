"""Index every configuration, failure, source receipt, and measurement class."""
from .common import *
from .analyze import table,pct


def main():
    initialize();rows=[];memory=[];host=[];contracts=[]
    import sys
    import importlib.metadata
    import torch
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()
        if any(v in d.metadata.get('Name','').lower() for v in ('torch','nixl','vllm','triton','cuda-cupti'))}
    write(ROOT/'software_versions.json',dict(Python=sys.version,Python_executable=sys.executable,
        Torch=torch.__version__,Torch_CUDA=torch.version.cuda,packages=packages,
        collector_headers_and_link_search='/usr/local/cuda-12.8',
        per_worker_resolved_CUPTI_library_map='NOT_RECORDED',
        private_runtime_HEAD=json.loads((ROOT/'state/preservation.json').read_text())['private_HEAD'],
        scope='CPU metadata inspection after scientific jobs. No GPU memory or model parameter change.'))
    for file in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(file.read_text())
        if 'name' not in cfg:continue
        folder=ROOT/'raw'/cfg['name'];receipt=folder/'dispatcher_result.json'
        dispatch=json.loads(receipt.read_text()) if receipt.exists() else {}
        qualification=folder/'qualification.json'
        q=json.loads(qualification.read_text()) if qualification.exists() else {}
        for path in sorted(folder.glob('native_contract_*.json')):
            value=json.loads(path.read_text())
            contracts.append(dict(name=cfg['name'],file=path.name,**{k:value.get(k) for k in
                ('role','phase','model_revision','fixed_depth','dtype','chunk_size','prefix_caching_enabled','reused_prefix_tokens')}))
        sources=[p for p in folder.glob('*source*') if p.is_file()]
        rows.append(dict(name=cfg['name'],model=cfg.get('model'),phase=cfg.get('phase'),
            count=cfg.get('count'),length=cfg.get('length'),cap=cfg.get('cap'),
            QPS=cfg.get('arrival_rate'),arrival_seed=cfg.get('arrival_seed'),
            physical_pair=cfg.get('gpus'),observer=cfg.get('idle_observer'),
            state_probe=cfg.get('state_probe'),calibration_serial=cfg.get('calibration_serial'),
            config_SHA256=digest(file),source_archive_SHA256={p.name:digest(p) for p in sources},
            dataset=cfg.get('dataset'),dataset_SHA256=digest(ROOT/'data'/cfg['dataset'])
                if cfg.get('dataset') and (ROOT/'data'/cfg['dataset']).exists() else None,
            dispatch_status=dispatch.get('status','RAW_WITHOUT_RECEIPT' if folder.exists() else 'NOT_RUN'),
            complete=dispatch.get('complete',False),dispatch_seconds=dispatch.get('elapsed_seconds'),
            exact_state_qualified=q.get('passed','NOT_QUALIFIED'),
            scope='All configurations remain listed. A complete dispatch is not a Safe Slack or performance qualification.'))
        window=folder/'window.json'
        if window.exists() and cfg.get('observer_variant'):
            bounds=json.loads(window.read_text());start,end=bounds['start'],bounds['end']
            def role_for(pid):
                for stem in ('memory_progress_light_','memory_progress_'):
                    sample=folder/f'{stem}{pid}.jsonl'
                    if sample.exists():
                        with sample.open() as stream:first=stream.readline()
                        if first:return json.loads(first).get('role','UNKNOWN_ROLE')
                return 'UNKNOWN_ROLE'
            for path in sorted(folder.glob('python_gc_*.json')):
                value=json.loads(path.read_text());events=[r for r in value['events'] if r['start']<end and r['end']>start]
                host.append(dict(name=cfg['name'],variant=cfg['observer_variant'],kind='GC',pid=value['pid'],role=role_for(value['pid']),
                    measured_events=len(events),p95_ms=pct([1000*(r['end']-r['start']) for r in events],95),
                    max_ms=max((1000*(r['end']-r['start']) for r in events),default=None),
                    scope='Measured cohort only. Startup is excluded. Temporal overlap is not causal attribution.'))
            for path in sorted(folder.glob('host_pauses_*.json')):
                value=json.loads(path.read_text());events=[r for r in value['events'] if r['start']<end and r['end']>start]
                host.append(dict(name=cfg['name'],variant=cfg['observer_variant'],kind='HOST_CALL_GE_20MS',pid=value['pid'],role=role_for(value['pid']),
                    measured_events=len(events),p95_ms=pct([1000*(r['end']-r['start']) for r in events],95),
                    max_ms=max((1000*(r['end']-r['start']) for r in events),default=None),
                    scope='Long nested host call enclosures. They are not additive CPU time or proof of GPU starvation.'))
        if cfg['name'] not in ('a_ouro_heavy_off_minimal_pilot_n128','a_huginn_heavy_off_minimal_pilot_n128'):continue
        # Keep allocator totals separate from their included component allocations.
        sample=json.loads((folder/'before_measured.json').read_text())
        total=81559*1024**2
        for key,values in sample.items():
            role='decode' if '-decode-' in key else 'prefill'
            for field in ('weight_bytes','reserved_kv_bytes','peak_allocated_bytes'):
                b=values[field]
                memory.append(dict(model=cfg['model'],role=role,component=field,bytes=b,GiB=b/1024**3,
                    percent_NVML_HBM=100*b/total,total_NVML_HBM_GiB=total/1024**3,
                    scope='Startup snapshot. peak_allocated_bytes includes component allocations and is not an additional stack component.'))
    table(ROOT/'configuration_inventory.csv',rows);write(ROOT/'configuration_inventory.json',rows)
    table(ROOT/'HBM_components.csv',memory);table(ROOT/'measured_host_observer_intervals.csv',host)
    table(ROOT/'runtime_contract_inventory.csv',contracts)
    write(ROOT/'inventory_status.json',dict(configurations=len(rows),complete=sum(r['complete'] for r in rows),
        failed=sum(r['dispatch_status'] not in ('COMPLETE','NOT_RUN','RAW_WITHOUT_RECEIPT') for r in rows),
        elapsed_dispatch_seconds=sum(r['dispatch_seconds'] or 0 for r in rows),
        active_budget='Fixed wall clock includes the recorded controller idle interval. Dispatch seconds are not GPU-kernel seconds.'))
    print(json.dumps(dict(configurations=len(rows),complete=sum(r['complete'] for r in rows))))


if __name__=='__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('configuration_and_memory_inventory'):main()
