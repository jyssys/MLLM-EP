"""Keep isolated operator models separate from online host identities."""
import json
import time
from .common import ROOT, write, digest
from .host_paths import analyze


def operator_lookup():
    gemms = {}
    attention = {}
    provenance = []
    for name in ('isolated_BF16_axis_geometry_v1', 'isolated_realized_axis_geometry_v1',
                 'isolated_realized_all_pair_geometry_v1',
                 'isolated_serving_projection_API_20261118_v1'):
        path = ROOT/'raw'/name/'operator_geometry.json'
        receipt = path.parent/'dispatcher_result.json'
        if not path.exists() or not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            continue
        value = json.loads(path.read_text())
        actual_API = name == 'isolated_serving_projection_API_20261118_v1'
        if actual_API:
            audit = json.loads((path.parent/'projection_API_audit.json').read_text())
            assert audit['actual_isolated_projection_API'].endswith('.linear_batch_invariant')
        provenance.append(dict(name=name, SHA256=digest(path),
            actual_Serving_projection_API=actual_API,
            source='Original layer0 weights, repeated captured activation rows; legacy FA3 operands are synthetic'))
        for row in value['results']:
            if row['kind'] == 'GEMM' and actual_API:
                gemms[row['GPU'], row['projection'], row['M']] = (row['p50_ms'], name)
            elif row['kind'] == 'ATTENTION':
                attention[row['GPU'], row['batch'], row['query_tokens'], row['KV_tokens']] = (row['p50_ms'], name)
    return gemms, attention, provenance


def shape_models(cfg, gemms, attention):
    folder = ROOT/'raw'/cfg['name']
    projections = ['model.layers.0.'+ending for ending in
                   ('self_attn.qkv_proj','self_attn.o_proj','mlp.gate_up_proj','mlp.down_proj')]
    output = {}
    for gpu, role in enumerate(('prefill', 'decode')):
        shapes = json.loads((folder/f'measured_geometry_{role}.json').read_text())['counts']
        gemm_total = attention_total = 0.
        missing_gemm, missing_attention = [], []
        used = set()
        inputs = 0
        for serialized, repetitions in shapes.items():
            shape = json.loads(serialized)
            if not shape['prefilling']:
                continue
            inputs += repetitions
            assert shape['requests'] == 1
            layers = ((36-cfg['static_layers']) if role=='prefill' else cfg['static_layers']) if cfg['evaluation_arm']=='Layer' else 36
            m = shape['padded_query_tokens']
            values = [gemms.get((gpu, projection, m)) for projection in projections]
            if any(v is None for v in values):
                missing_gemm.append(dict(M=m, count=repetitions))
            else:
                gemm_total += sum(v[0] for v in values)*layers*repetitions
                used.update(v[1] for v in values)
            # This is a nonpaged shape model, not the serving Attention time.
            q = shape['query_tokens']
            k = q if role=='prefill' else 8000
            value = attention.get((gpu, 1, q, k))
            if value is None:
                missing_attention.append(dict(Q=q, K=k, count=repetitions))
            else:
                attention_total += value[0]*layers*repetitions
                used.add(value[1])
        output[role] = dict(observed_prefill_model_inputs=inputs,
            modeled_four_projection_ms_per_request=None if missing_gemm else gemm_total/cfg['count'],
            modeled_contiguous_FA3_ms_per_request=None if missing_attention else attention_total/cfg['count'],
            missing_GEMM_shapes=missing_gemm, missing_attention_shapes=missing_attention,
            isolated_sources=sorted(used), evidence='STRUCTURAL_ACCOUNTING',
            scope='Use only the explicit Serving linear_batch_invariant API for projection models. The legacy H100 torch.nn.functional.linear controls are not substituted. Scale isolated layer0 event medians by actual shape repetitions and layer counts. These are modeled sums, not online GPU measurements. Contiguous FA3 is not paged serving Attention. No Norm, Activation, host or transfer cost is included.')
    return output


def main():
    design = json.loads((ROOT/'state/batch_matched_design_20261118.json').read_text())
    gemms, attention, provenance = operator_lookup()
    rows = []
    for name in design['names']:
        folder = ROOT/'raw'/name
        gate = ROOT/'analysis'/(name+'_matched_batch_gate.json')
        if not gate.exists():
            continue
        assert json.loads(gate.read_text())['passed']
        cfg = json.loads((folder/'configuration.json').read_text())
        value = json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
        path = ROOT/'analysis'/(name+'_host_paths.json')
        host = json.loads(path.read_text()) if path.exists() else analyze(name)
        spans = {k:v['mean'] for k,v in host['summaries'].items()}
        assert abs(sum(spans.values())-value['mean_TTFT_ms']) < 1e-5
        transfer = value['transport_audit']
        assert transfer['KV_byte_gate'] and transfer['hidden_byte_gate']
        rows.append(dict(name=name, stage=cfg['matrix_stage'], method=cfg['evaluation_arm'],
            layers=cfg['static_layers'], output=cfg['cap'], count=cfg['count'],
            QPS=cfg['arrival_rate'], seed=cfg['arrival_seed'],
            mean_TTFT_ms=value['mean_TTFT_ms'], TTFT_p95_ms=value['TTFT_p95_ms'],
            exact_host_mean_TTFT_identity_ms=spans,
            D_preparation_host_ms_per_request=host['summed_cohort_preparation_per_completed_request_ms'],
            D_decode_residency_host_ms_per_request=host['summed_cohort_decode_residency_per_completed_request_ms'],
            actual_KV_bytes_per_request=transfer['NIXL_reported_KV_bytes']/cfg['count'],
            actual_hidden_residual_bytes_per_request=transfer['NIXL_reported_hidden_residual_bytes']/cfg['count'],
            transfer_telemetry=transfer, isolated_operator_models=shape_models(cfg, gemms, attention),
            exclusive_online_GEMM_time='NOT_MEASURED', exclusive_online_attention_time='NOT_MEASURED',
            online_SM_HBM_counters='NOT_MEASURED', P_compute_saturation='NOT_IDENTIFIED',
            scope='Online host identities and measured byte counters stay separate from isolated operator models. Component p95 values do not decompose TTFT p95. Decode residency is an elapsed worker-notice span, not exclusive GPU busy. Do not subtract a modeled GEMM sum from a host enclosure and call the remainder Attention or Runtime overhead.'))
    write(ROOT/'analysis/batch1_cost_and_critical_path.json', dict(epoch=time.time(),
        starts=rows, isolated_provenance=provenance,
        scope='Prospective batch-matched campaign. Interpret causal mechanisms only with the matched geometry gate and independent-start latency results. A crossover is not assumed.'))
    print('BATCH1_COST_AND_HOST_IDENTITIES', len(rows), flush=True)


if __name__ == '__main__':
    main()
