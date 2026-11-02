"""Distinguish NVML total HBM from CUDA-visible managed capacity."""
from .common import *
from .analyze import records,table


def main():
    import pynvml
    pynvml.nvmlInit()
    totals=[]
    for index in range(8):
        handle=pynvml.nvmlDeviceGetHandleByIndex(index)
        value=pynvml.nvmlDeviceGetMemoryInfo(handle)
        totals.append(dict(physical_GPU=index,UUID=pynvml.nvmlDeviceGetUUID(handle),NVML_total_bytes=value.total,
            NVML_total_GiB=value.total/1024**3,time=time.time()))
    assert len({v['NVML_total_bytes'] for v in totals})==1
    physical=totals[0]['NVML_total_bytes'];rows=[]
    for folder in (ROOT/'raw').iterdir():
        if not folder.is_dir():continue
        for p in folder.glob('memory_inventory_*.json'):
            inventory=json.loads(p.read_text());pid=p.stem.rsplit('_',1)[1]
            samples=records(folder/f'memory_progress_{pid}.jsonl')
            if not samples:continue
            sample=min(samples,key=lambda r:abs(r['time']-inventory['time']))
            cuda=sample['device_total_bytes']
            row=dict(name=folder.name,role=inventory['role'],endpoint='INITIAL' if '_initial_' in p.name else 'CLOSE',
                inventory_time=inventory['time'],nearest_snapshot_delta_s=sample['time']-inventory['time'],
                NVML_total_bytes=physical,CUDA_visible_total_bytes=cuda,
                non_CUDA_managed_baseline_bytes=physical-cuda,
                CUDA_free_bytes=sample['device_free_bytes'],
                torch_allocated_bytes=sample['allocator_allocated_bytes'],torch_reserved_bytes=sample['allocator_reserved_bytes'],
                cached_not_allocated_bytes=sample['allocator_reserved_bytes']-sample['allocator_allocated_bytes'],
                outside_torch_reservation_including_driver_bytes=physical-sample['device_free_bytes']-sample['allocator_reserved_bytes'],
                **{k:v for k,v in inventory.items() if k.endswith('_bytes')})
            for k,v in list(row.items()):
                if k.endswith('_bytes') and isinstance(v,(int,float)):
                    row[k+'_GiB']=v/1024**3
                    row[k+'_percent_NVML_total_HBM']=100*v/physical
                    row[k+'_percent_CUDA_visible_capacity']=100*v/cuda
            row.update(evidence='MEASURED_OWNER_STORAGE_AND_CUDA_SNAPSHOT_WITH_READ_ONLY_NVML_TOTAL',
                scope='NVML total is static for this device. Other values belong to the named owner snapshot. Earlier percent_HBM columns use CUDA-visible capacity, not NVML total. Do not add allocator totals to contained tensor categories.')
            rows.append(row)
    write(ROOT/'state/memory_denominator_audit.json',dict(time=time.time(),devices=totals,
        scope='H100 NVML total and CUDA-visible capacity differ by the non-CUDA-managed baseline. Both denominators remain explicit. No original raw memory record is changed.'))
    table('physical_hbm_accounting.csv',rows)
    print(json.dumps(dict(owner_endpoints=len(rows),NVML_total_GiB=physical/1024**3)))


if __name__=='__main__':main()
