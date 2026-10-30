"""Report measured worker memory separately from structural slab sizes."""
import json
from .common import ROOT
from .analyze import csv_write


def main():
    rows=[]
    for folder in sorted((ROOT/'raw').glob('map_*')):
        if not (folder/'after_measured.json').exists():continue
        cfg=json.loads((folder/'arc_contract.json').read_text())
        stats=json.loads((folder/'after_measured.json').read_text())
        mapping=json.loads((folder/'worker_map.json').read_text())
        width=2048 if cfg['model']=='ouro' else 10560
        for worker,s in stats.items():
            role=mapping[worker]['role'];slots=cfg['capacity']+(8 if role=='prefill' else 32)
            hidden_rows=1 if cfg.get('handoff_mode')=='native_G1' else cfg['state_capacity_tokens']
            rows.append(dict(name=folder.name,model=cfg['model'],role=role,
                             physical_GPU=mapping[worker]['physical_GPU'],uuid=mapping[worker]['uuid'],
                             measured_weight_bytes=s['weight_bytes'],measured_reserved_KV_bytes=s['reserved_kv_bytes'],
                             measured_peak_Torch_allocated_bytes=s['peak_allocated_bytes'],
                             handoff_mode=cfg.get('handoff_mode','cut_snapshot'),
                             registered_hidden_slab_bytes_structural=slots*hidden_rows*width*2,
                             runner_prompt_state_bytes_structural=cfg['capacity']*cfg['state_capacity_tokens']*width*2,
                             graph_workspace_bytes='NOT_MEASURED',
                             scope='Peak since process start. Not a reset per-cut peak. Slab sizes are structural allocations.',
                             matched_control_limit='Full-PD control keeps continuation capacity. A dedicated full-PD deployment could reserve less hidden HBM.'))
    csv_write('worker_memory.csv',rows)
    print('WORKER_MEMORY_ROWS',len(rows))


if __name__=='__main__':main()
