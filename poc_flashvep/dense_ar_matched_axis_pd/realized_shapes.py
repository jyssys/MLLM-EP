"""Measure the actual padded GEMM row shapes from completed pilots."""
import argparse
import json
import time
from .common import ROOT,save,write
from .configuration import base
from .run import execute


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-pairs',action='store_true')
    args=parser.parse_args()
    while not (ROOT/'state/probe_complete.json').exists():
        time.sleep(15)
    shapes={8000}
    sources=[]
    pattern='discovery_*_N144_O200_q2_s2026111780_A8_v3.json' if args.all_pairs else 'probe_*.json'
    paths=sorted((ROOT/'configs').glob(pattern))
    if args.all_pairs:
        assert len(paths)==8,paths
        while not all((ROOT/'raw'/path.stem/'dispatcher_result.json').exists() for path in paths):
            time.sleep(15)
    for path in paths:
        cfg=json.loads(path.read_text())
        if cfg['arm']!='PD_G1' or cfg.get('execution_contract_version')!=3:
            continue
        receipt=json.loads((ROOT/'raw'/cfg['name']/'dispatcher_result.json').read_text())
        assert receipt['complete'],cfg['name']
        sources.append(cfg['name'])
        for role in ('prefill','decode'):
            geometry=json.loads((ROOT/'raw'/cfg['name']/f'measured_geometry_{role}.json').read_text())['counts']
            shapes.update(json.loads(key)['padded_query_tokens'] for key in geometry)
    assert len(sources)==(7 if args.all_pairs else 6),sources
    assert shapes and all(0<m<=8192 for m in shapes)
    stem='isolated_realized_all_pair_geometry_v1' if args.all_pairs else 'isolated_realized_axis_geometry_v1'
    cfg=base(stem,'Native',0,8,16,4.,2026111780,
        'heldout128_8000.json',phase='ISOLATED_REALIZED_OPERATOR_DIAGNOSTIC',
        auxiliary_only=True,entry_module='poc_flashvep.dense_ar_matched_axis_pd.isolated',
        wall_timeout=1500,isolated_widths=sorted(shapes),isolated_skip_attention=False)
    design='realized_all_pair_GEMM_shape_design.json' if args.all_pairs else 'realized_GEMM_shape_design.json'
    write(ROOT/'analysis'/design,dict(names=sources,
        widths=sorted(shapes),evidence='TRACE_ANALYSIS',
        scope='Measure every padded row count observed in the fixed paired pilots. Selection does not change the serving partitions or scheduler. This includes Decode shapes. Operator times remain isolated diagnostics.'))
    assert execute(save(cfg)),cfg['name']
    flag='realized_shapes_all_pairs_complete.json' if args.all_pairs else 'realized_shapes_complete.json'
    write(ROOT/'state'/flag,dict(time=time.time(),name=cfg['name']))


if __name__=='__main__':
    main()
