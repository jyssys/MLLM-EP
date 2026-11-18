"""Prepare explicit real-document pools and immutable qualification inputs."""
import argparse
import json
from .common import ROOT, PREVIOUS, initialize, save, write, digest
from .configuration import base


def pools():
    initialize()
    contract=ROOT/'configs/common_output_sweep_SLO.json'
    if not contract.exists():
        import time
        write(contract,dict(frozen_epoch=time.time(),TTFT_ms=5000,ITL_ms=34,TPOT_ms=33,
            source='User-specified historical Output200 Balanced thresholds',
            role='SECONDARY_COMMON_THRESHOLD_OUTPUT_SWEEP_CONTROL',
            scope='Declared before any clean candidate performance. Apply the same thresholds to all five outputs. Keep every original per-output frozen SLO unchanged. This secondary view separates output-length effects from the historical Output64 Balanced ITL35ms threshold.'))
    target=ROOT/'data/pool_contract.json'
    if target.exists():
        return
    discovery=json.loads((ROOT/'data/discovery.json').read_text())
    heldout=json.loads((ROOT/'data/heldout128_8000.json').read_text())
    # Arithmetic requires a multiple of9. Reuse only declared real documents.
    # This campaign studies controlled fixed partitions, not new-data tuning.
    # Preserve the original128 unique confirmation documents and append16
    # distinct old discovery documents. This is not a wholly unseen144 pool.
    ids={r['prompt_id'] for r in heldout}
    extra=[r for r in discovery if r['prompt_id'] not in ids][:16]
    assert len(extra)==16 and len({r['prompt_id'] for r in heldout+extra})==144
    write(ROOT/'data/matched144.json', heldout+extra)
    repeated=[discovery[i%len(discovery)] for i in range(72)]
    write(ROOT/'data/discovery72_declared_cycle.json', repeated)
    write(target,dict(discovery_count=72,discovery_unique=len({r['prompt_id'] for r in repeated}),
        confirmation_count=144,confirmation_unique=144,
        confirmation_new_in_previous_campaign=128,confirmation_old_discovery_documents=16,
        source_files=[dict(path=str(PREVIOUS/'data'/name),SHA256=digest(PREVIOUS/'data'/name))
                      for name in ('discovery.json','heldout128_8000.json')],
        scope='Existing frozen real documents. No filler. No globally unseen claim. Exact matched rows require N divisible by9. Fresh arrival seeds are separate from historical starts.'))


def historical_outputs():
    from .common import foundation
    target=ROOT/'raw/HISTORICAL_original_outputs/requests.jsonl'
    if target.exists():
        return
    sources=[PREVIOUS/'raw/unseen_Native_reference_N128_O200_repeat1_v1/requests.jsonl',
        foundation.PREVIOUS/'raw/baseline_Native_O200_N64_q4p8_s21_v1/requests.jsonl']
    rows={}
    for path in sources:
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['prompt_id'] in rows:
                assert rows[row['prompt_id']]['output_ids']==row['output_ids']
            rows[row['prompt_id']]=row
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:
        for row in rows.values():
            stream.write(json.dumps(row)+'\n')
    write(ROOT/'references/historical_output_provenance.json',dict(
        evidence='HISTORICAL_CORRECTNESS_REFERENCE_ONLY',
        sources=[dict(path=str(p),SHA256=digest(p)) for p in sources],
        rows=len(rows),copy_SHA256=digest(target),
        scope='Only greedy outputs and prompt identity are used. Historical latency is not a new measurement.'))


def qualification(method, count=8, layers=12, version=1):
    pools()
    historical_outputs()
    name=f'qual_{method}_Last{layers if method in ("Layer","Token") else 0}_N{count}_v{version}'
    cfg=base(name,method=method,layers=layers,count=count,cap=16,rate=4.,
        dataset='heldout128_8000.json',correctness=True,
        phase='UNIFIED_FULL_STATE_QUALIFICATION',
        arrival_offsets=[0.]*count,arrival_rate=None,
        warmup_count=0,warmup_batch_sizes=[],negotiation_timeout_s=120.,
        runtime_observation=True,verify_layer_execution=True,
        witness_hidden_partition_tokens=[880,896,1776,1792,2656,2672],
        axis_state_save_tensors=False)
    cfg['reference_name']='HISTORICAL_original_outputs'
    if (ROOT/'state/design_freeze_v3_before_performance.json').exists():
        cfg.update(common_prospective_KV_reservation=method!='Agg',execution_contract_version=3)
    if method=='Native':
        cfg.update(axis_state_witness=True,verify_native_prefix_KV=False,
            imported_native_state_witness=True)
    elif method=='Token':
        cfg.update(axis_state_witness=True,layer_mode=None,hidden_transport=None)
        lo,hi={4:(880,896),8:(1776,1792),12:(2656,2672)}[layers]
        cfg['token_suffix_by_request']=[(lo,hi)[i%2] for i in range(count)]
    elif method=='Layer':
        cfg.update(full_prompt_state_witness=True,hidden_verify_hash=True,
            full_prompt_state_witness_roles=['prefill','decode'])
    else:
        cfg.update(full_prompt_state_witness=True,post_graph_state_witness=True,
            full_prompt_state_witness_roles=['replica0','replica1'])
    return cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',default='Native',choices=['Native','Agg','Layer','Token'])
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--layers',type=int,default=12)
    p.add_argument('--version',type=int,default=1)
    args=p.parse_args()
    print(save(qualification(args.method,args.count,args.layers,args.version)))


if __name__=='__main__':
    main()
