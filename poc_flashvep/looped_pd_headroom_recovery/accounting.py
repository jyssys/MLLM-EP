"""Keep physical storage, coordinator credit, valid frontiers, and work distinct."""
import math
from .common import *
from .analyze import records,table


def main():
    budgets=[];envelopes=[];lifetimes=[];projection=[]
    for path in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or 'name' not in cfg:continue
        folder=ROOT/'raw'/cfg['name']
        if not folder.exists():continue
        window=json.loads((folder/'window.json').read_text()) if (folder/'window.json').exists() else None
        snapshots=[]
        for p in folder.glob('memory_progress_*.jsonl'):snapshots.extend(records(p))
        for p in folder.glob('memory_inventory_initial_*.json'):
            inventory=json.loads(p.read_text());role=inventory['role']
            samples=[r for r in snapshots if r['role']==role]
            if not samples:continue
            sample=min(samples,key=lambda r:abs(r['time']-inventory['time']))
            total=sample['device_total_bytes']
            row=dict(name=cfg['name'],model=cfg['model'],role=role,time=inventory['time'],
                closest_live_snapshot_delta_s=sample['time']-inventory['time'],
                actual_device_total_bytes=total,device_free_bytes=sample['device_free_bytes'],
                allocator_cached_not_allocated_bytes=sample['allocator_reserved_bytes']-sample['allocator_allocated_bytes'],
                device_used_outside_torch_reserved_bytes=total-sample['device_free_bytes']-sample['allocator_reserved_bytes'],
                **{k:v for k,v in inventory.items() if k.endswith('_bytes')},
                evidence='MEASURED_UNIQUE_STORAGE_AND_NEARBY_OWNER_CUDA_SNAPSHOT',
                scope='Each row belongs to one worker and timestamp. Opaque allocator bytes include unexposed graph/library storage. Allocator totals overlap component storage and are not stack additions.')
            for k,v in list(row.items()):
                if k.endswith('_bytes') and isinstance(v,(int,float)):
                    row[k+'_GiB']=v/1024**3;row[k+'_percent_HBM']=100*v/total
            budgets.append(row)
        if snapshots:
            allocated_per_token=786432 if cfg['model']=='ouro' else 5406720
            blocks=cfg['depth']*math.ceil((cfg['length']+cfg['cap']-1)/16)
            for role in ('prefill','decode'):
                samples=[r for r in snapshots if r['role']==role]
                if not samples:continue
                first=samples[0]
                required=cfg['depth']*math.ceil((cfg['length']+(cfg['cap']-1 if role=='decode' else 0))/16)
                measured=[r for r in samples if window is None or window['start']<=r['time']<=window['end']]
                envelopes.append(dict(name=cfg['name'],model=cfg['model'],role=role,length=cfg['length'],cap=cfg['cap'],
                    input_output_ratio=cfg['length']/cfg['cap'],required_native_blocks=required,
                    per_request_allocated_bytes=required*first['KV_block_bytes'],
                    per_request_allocated_GiB=required*first['KV_block_bytes']/1024**3,
                    per_request_percent_HBM=100*required*first['KV_block_bytes']/first['device_total_bytes'],
                    per_request_percent_KV_pool=100*required/first['KV_total_blocks'],
                    structural_KV_count_before_slots=first['KV_total_blocks']//required,
                    configured_compute_slots=cfg.get('P_slots' if role=='prefill' else 'D_slots'),
                    observed_owner_requests_max=max((len(r['requests']) for r in measured),default=None),
                    observed_active_compute_phase_blocks_max=max((r.get('KV_active_request_blocks',0) for r in measured),default=None)
                        if measured and all('KV_active_request_blocks' in r for r in measured) else None,
                    evidence='STRUCTURAL_NATIVE_GEOMETRY_PLUS_MEASURED_OWNER_COUNTS',
                    scope='The structural floor excludes slots, growth watermark, pending reservations, and timing. It is not guaranteed serving concurrency. Active allocation is not kernel busy.'))
        lifetimes.extend(dict(name=cfg['name'],model=cfg['model'],**r) for r in records(folder/'lifetime_events.jsonl'))
        for p in folder.glob('lifetime_owner_*.jsonl'):
            lifetimes.extend(dict(name=cfg['name'],model=cfg['model'],**r) for r in records(p))
        for p in folder.glob('projection_patch_*.json'):
            record=json.loads(p.read_text())
            if record.get('geometry')!=256:continue
            for batch in (1,2,4,8):
                projection.append(dict(name=cfg['name'],model=cfg['model'],batch=batch,
                    qualified_variant=record['variant'],selected_physical_linear_modules=len(record['selected']),
                    selected_linear_decode_GEMM_rows=256,real_decode_rows=batch,
                    selected_linear_row_padding_factor=256/batch,
                    prefill_rows=cfg['length'],prefill_tiled_rows=256*math.ceil(cfg['length']/256),
                    prefill_selected_linear_padding_factor=256*math.ceil(cfg['length']/256)/cfg['length'],
                    evidence='SOURCE_VERIFIED_STRUCTURAL_SELECTED_PROJECTION_ACCOUNTING',
                    scope='Applies only to selected Down/adapter Linear calls, not whole-model FLOPs or measured speedup. The qualified numerical patch remains enabled in every comparison.'))
            break
    table('gpu_memory_inventory.csv',budgets)
    table('workload_capacity_envelope.csv',envelopes)
    table('admission_ack_release_timeline.csv',lifetimes)
    table('qualified_projection_geometry_accounting.csv',projection)
    print(json.dumps(dict(memory_budgets=len(budgets),workload_rows=len(envelopes),native_lifetime_events=len(lifetimes))))


if __name__=='__main__':main()
