"""Queue one bounded projection-path control under the existing GPU lease."""
import json
import time
from .common import ROOT, save, write, digest
from .configuration import matched_token_plan
from .run import execute


NAME = 'isolated_serving_projection_API_20261118_v1'


def summarize():
    path = ROOT/'raw'/NAME/'operator_geometry.json'
    result = json.loads(path.read_text())
    sums = {}
    for row in result['results']:
        key = (row['GPU'], row['M'])
        sums[key] = sums.get(key, 0.)+row['p50_ms']
    rows = []
    for gpu in (0, 1):
        for layers in (4, 8, 12):
            suffix = matched_token_plan(layers, 72, group_size=1)
            layer = layers*sums[gpu, 8000]
            token = 36*sum(sums[gpu, s] for s in suffix)/len(suffix)
            rows.append(dict(GPU=gpu, layers=layers, matched_token_rows=8000*layers,
                modeled_Layer_projection_sum_ms=layer, modeled_Token_projection_sum_ms=token,
                modeled_Token_over_Layer=token/layer,
                evidence='STRUCTURAL_ACCOUNTING',
                scope='Scale isolated layer0 projection medians to a matched token-layer row plan. This is not a full-model measurement, TTFT, TPOT or an exclusive kernel sum. Different layer weights, cache residency and online launch timing can change actual cost.'))
    write(ROOT/'analysis/actual_projection_API_geometry_model.json',dict(epoch=time.time(),
        source=NAME, source_SHA256=digest(path), comparisons=rows,
        source_operator_evidence='MEASURED_ISOLATED_CUDA_EVENT_ENCLOSURES',
        model_evidence='STRUCTURAL_ACCOUNTING',
        historical_operator_limit='The historical H100 F.linear benchmarks use another call path and another start. Do not treat their timing difference as a controlled estimate of the API effect.',
        online_exclusive_GEMM_time='NOT_MEASURED'))
    return rows


def configuration():
    cfg = json.loads((ROOT/'configs/isolated_BF16_axis_geometry_v1.json').read_text())
    cfg.update(name=NAME, phase='ISOLATED_SERVING_PROJECTION_API_CONTROL',
        matrix_stage='isolated_serving_projection_API',
        entry_module='poc_flashvep.dense_ar_matched_axis_pd.isolated_serving_gemm',
        isolated_skip_attention=True, isolated_widths=[1,2,3,4,432,448,880,896,
            1776,1792,2656,2672,5328,5344,6208,6224,7104,7120,8000],
        actual_isolated_projection_API='linear_batch_invariant',
        auxiliary_only=True, wall_timeout=900)
    return cfg


def main():
    cfg = configuration()
    design = ROOT/'state/serving_projection_API_control_design.json'
    value = dict(name=NAME, projection_API='linear_batch_invariant',
        predecessor_API='torch.nn.functional.linear', widths=cfg['isolated_widths'],
        evidence='NOT_RUN', scope='One isolated operator control. Same original BF16 weights and captured activation rows. No simultaneous GPU work. Do not count this as Online Serving or a Model Forward benchmark.')
    if not design.exists():
        write(design, dict(epoch=time.time(), **value))
    try:
        assert execute(save(cfg)), NAME
        data = json.loads((ROOT/'raw'/NAME/'operator_geometry.json').read_text())
        audit = json.loads((ROOT/'raw'/NAME/'projection_API_audit.json').read_text())
        assert audit['actual_isolated_projection_API'].endswith('.linear_batch_invariant')
        assert len(data['results']) == 2*4*len(cfg['isolated_widths'])
        assert all(r['kind']=='GEMM' and r['dtype']=='torch.bfloat16' for r in data['results'])
        summarize()
        write(ROOT/'state/serving_projection_API_control_complete.json',dict(epoch=time.time(),
            passed=True, name=NAME, results=len(data['results']),
            scope='Isolated actual Projection API event samples. Scaling one layer to multiple layers remains a cost model, not measured Online GEMM time.'))
    except BaseException as error:
        write(ROOT/'state/serving_projection_API_control_stopped.json',dict(epoch=time.time(),error=repr(error)))
        raise


if __name__ == '__main__':
    main()
