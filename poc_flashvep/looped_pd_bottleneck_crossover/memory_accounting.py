"""Exact allocation geometry is not a measured cache-optimization speedup."""
import math
from .common import *
from .prepare import GRID
from .summary_tables import save_csv

def main():
    rows=[]
    for model in ('ouro','huginn'):
        paths=list((ROOT/'raw').glob(f'phase_a_{model}*/geometry_*.json'))
        seen=set()
        for p in paths:
            g=json.loads(p.read_text());pool=g['KV_blocks']
            if pool in seen:continue
            seen.add(pool);layers=g['layers'];R=g['storage_depths'];block=16
            recurrent=len(g.get('recurrent_kv_layers',list(range(layers))))
            valid_planes=recurrent*R+(layers-recurrent)
            dense_planes=layers*R;fraction=valid_planes/dense_planes
            for cell,(length,cap) in GRID.items():
                tokens=length+cap-1;blocks=math.ceil(tokens/block)*R
                reserved=blocks*g['KV_block_bytes'];minimum=reserved*fraction
                rows.append(dict(model=model,cell=cell,input_tokens=length,output_tokens=cap,
                    physical_layers=layers,recurrent_layers=recurrent,depth=R,
                    allocated_layer_depth_planes=dense_planes,semantically_used_layer_depth_planes=valid_planes,
                    unused_boundary_depth_plane_fraction=1-fraction,pool_blocks=pool,pool_bytes=g['KV_bytes'],
                    actual_full_reservation_bytes_per_request=reserved,
                    actual_pool_request_limit=pool//blocks,
                    hypothetical_compact_KV_bytes_per_request=minimum,
                    hypothetical_compact_pool_request_limit=int(g['KV_bytes']//minimum),
                    valid_KV_bytes_per_token=(g['KV_block_bytes']/block)*R*fraction,
                    allocated_KV_bytes_per_token=(g['KV_block_bytes']/block)*R,
                    evidence='STRUCTURAL_ACCOUNTING_FROM_MEASURED_ALLOCATION_GEOMETRY',
                    scope='Compaction is NOT_IMPLEMENTED. This is not measured extra capacity or serving speedup. Native transfer already skips unused boundary-depth planes.',
                    source=str(p.relative_to(ROOT))))
    save_csv(ROOT/'KV_geometry_accounting.csv',rows)
    print('KV_GEOMETRY_ROWS',len(rows))

if __name__=='__main__':main()
