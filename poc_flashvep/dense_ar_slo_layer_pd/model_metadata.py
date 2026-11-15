"""Separate new empirical measurements from borrowed estimates and roofline floors."""
from .common import *
from .coalesced_costs import parse


def main():
    old=json.loads((ROOT/'configs/coalesced_cost_model_v1.json').read_text())
    names=old['calibration_names'];rows=[r for n in names for r in parse(n)]
    keys=('P_prefix_ms','handoff_ms','D_suffix_ms','D_commit_ms',
        'negotiation_idle_ms','negotiation_busy_ms','decode_step_p95_ms',
        'decode_step_by_batch_p95_ms','uncertainty')
    model={k:old[k] for k in keys}
    model.update(model_version='COALESCED_EMPIRICAL_V2_METADATA_CORRECTION',
        numerical_tables_changed=False,legacy_v1_SHA256=digest(ROOT/'configs/coalesced_cost_model_v1.json'),
        primitive_calibration_names=names,primitive_sample_counts={str(b):{str(l):sum(
            r['batch']==b and r['layers']==l for r in rows) for l in (0,2,4,8,12)} for b in (1,2,3,4,8,16)},
        primitive_files={name:{file:digest(ROOT/'raw'/name/file) for file in
            ('source.zip','actual_submissions.jsonl','actual_cut_commits.jsonl','native_handoffs.jsonl','CUDA_event_enclosures_decode.json')}
            for name in names},
        borrowed_components=dict(source='revised_cost_model_v3.json',
            SHA256=digest(ROOT/'configs/revised_cost_model_v3.json'),
            components=['negotiation','Decode-step Host lookup','initial uncertainty margins'],
            scope='These components were not recalibrated by the new suffix microbenchmarks. Online coverage remains a separate validation.'),
        future_confirmation_information=False,
        scope='Empirical stream-event enclosures and Host boundaries. Three samples per primitive cell do not establish p99 or WCET. Source v1 inherited stale ancestry fields. This version keeps the same numerical tables and identifies every measurement source.')
    target=ROOT/'configs/coalesced_cost_model_v2.json'
    if target.exists():assert json.loads(target.read_text())==model
    else:write(target,model)
    config=json.loads((MODEL/'config.json').read_text())
    h=config['hidden_size'];f=config['intermediate_size'];q=config['num_attention_heads']
    kv=config['num_key_value_heads'];dim=config['head_dim'];length=8000
    weights=h*(q+2*kv)*dim+h*q*dim+3*h*f
    evidence=[]
    for layers in (2,4,8,12,36):
        gemm=2*length*weights*layers
        causal_attention=2*layers*q*dim*length*(length+1)
        weight_bytes=2*weights*layers
        # This one-read traffic excludes activations and KV. It is an
        # optimistic floor, not a measured bandwidth or a complete traffic model.
        evidence.append(dict(layers=layers,prompt_tokens=length,
            projection_and_FFN_FLOPs=gemm,causal_attention_FLOPs=causal_attention,
            weight_bytes_once=weight_bytes,
            ideal_dense_GEMM_floor_ms=1000*gemm/(989.4e12),
            ideal_one_weight_read_floor_ms=1000*weight_bytes/(3.35e12)))
    write(ROOT/'analysis/hybrid_model_structural_accounting.json',dict(model_revision=MODEL_REVISION,
        model_config_SHA256=digest(MODEL/'config.json'),rows=evidence,
        dense_BF16_peak_TFLOPs_s=989.4,HBM_peak_TB_s=3.35,
        hardware_source='https://resources.nvidia.com/en-us-hopper-architecture',
        sparsity_used=False,evidence='STRUCTURAL_ACCOUNTING',
        limitations=['Peak specifications are not measured execution efficiency.',
            'Weight-only traffic omits activation and KV traffic.',
            'Attention, normalization, graph, Host, transfer and queue costs require empirical measurements.',
            'Online admission uses empirical completion predictions, not the roofline floor.']))


if __name__=='__main__':main()
