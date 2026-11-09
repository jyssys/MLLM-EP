"""Summarize measured memory samples without stacking overlapping totals."""
from .common import *
from .campaign_report import write_csv


def main():
    rows=[];NVML=81559*2**20
    for receipt in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        if not json.loads(receipt.read_text()).get('complete'):continue
        folder=receipt.parent;cfg=json.loads((ROOT/'configs'/(folder.name+'.json')).read_text())
        if not folder.name.startswith(('confirm_','final_calibration_','r1_')):continue
        begin=json.loads((folder/'measurement_begin.json').read_text())['time']
        for file in folder.glob('memory_progress_*.jsonl'):
            data=[]
            with file.open() as handle:
                for line in handle:
                    row=json.loads(line)
                    if row['time']>=begin:data.append(row)
            if not data:continue
            total=data[0]['device_total_bytes']
            for field,operation in [('device_free_bytes',min),('allocator_allocated_bytes',max),
                    ('allocator_reserved_bytes',max),('allocator_peak_bytes',max),('KV_active_request_allocated_bytes',max)]:
                amount=operation(r[field] for r in data)
                rows.append(dict(name=folder.name,model=cfg['model'],role=data[0]['role'],
                    evidence_class='STATE_DIAGNOSTIC' if cfg.get('state_probe') else 'MARKER_FREE_OR_OTHER_DIAGNOSTIC',
                    field=field,operation=operation.__name__,bytes=amount,GiB=amount/2**30,
                    NVML_HBM_percent=100*amount/NVML,CUDA_accessible_percent=100*amount/total,
                    minimum_free_KV_blocks=min(r['KV_free_blocks'] for r in data),
                    maximum_used_KV_blocks=max(r['KV_used_blocks'] for r in data),
                    total_KV_blocks=data[0]['KV_total_blocks'],samples=len(data),
                    scope='Measured-interval owner samples. Free memory uses cudaMemGetInfo. Allocated, reserved, and peak totals overlap. Active KV means compute-phase ownership, not executing kernels. Sampling can miss shorter peaks.'))
    write(ROOT/'memory_capacity_summary.json',rows)
    write_csv(ROOT/'memory_capacity_summary.csv',rows)
    print(json.dumps(dict(rows=len(rows))))


if __name__=='__main__':main()
