"""Report clean operator enclosures without inventing online GPU components."""
import collections
import json
import statistics
from .common import ROOT,write,digest
from ..dense_ar_layer_token_pd.token_axis import matched_token_plan


def main():
    name='isolated_actual_serving_API_Native_Last0_N36_O16_q2_s2026111904_B1_W32_v1'
    path=ROOT/'raw'/name/'operator_geometry.json'
    if not path.exists():
        print('OPERATOR_ANALYSIS_NOT_RUN',flush=True)
        return
    data=json.loads(path.read_text())
    rows=data['results'];profiles=data['profiles']
    by={(r['GPU'],r['projection'],r['M']):r for r in rows if r['kind']=='GEMM'}
    projections=sorted({r['projection'] for r in rows if r['kind']=='GEMM'})
    assert len(projections)==4
    lookup={(gpu,m):sum(by[gpu,p,m]['p50_ms'] for p in projections)
        for gpu in (0,1) for m in sorted({r['M'] for r in rows if r['kind']=='GEMM'})}
    comparison=[]
    for m in (2,4,8,12):
        suffix=matched_token_plan(m,72,group_size=1)
        L_D=m*lookup[1,8000]
        T_D=36*statistics.mean(lookup[1,s] for s in suffix)
        L_P=(36-m)*lookup[0,8000]
        T_P=36*statistics.mean(lookup[0,8000-s] for s in suffix)
        comparison.append(dict(layers=m,requests=72,suffix_histogram=dict(collections.Counter(suffix)),
            Layer_D_projection_enclosure_model_ms=L_D,Token_D_projection_enclosure_model_ms=T_D,
            Token_over_Layer_D_projection_model=T_D/L_D,
            Layer_P_projection_enclosure_model_ms=L_P,Token_P_projection_enclosure_model_ms=T_P,
            evidence='STRUCTURAL_ACCOUNTING',
            scope='Weighted sum of isolated original-layer0 projection medians across the planned layer counts. Valid token-layer rows match exactly. This is not a measured whole Forward, online exclusive GEMM time, attention time or Serving speedup.'))
    qualified=[];diagnostic=[]
    for p in profiles:
        item=dict(label=p['label'],observer_gate_5pct=p.get('observer_gate_5pct',False),
            observed_over_control=p.get('observed_over_control'),controls_within_5pct=p.get('controls_within_5pct'),
            kernels=p.get('kernels'),scope=p.get('scope'))
        if p.get('observer_gate_5pct'):
            item['evidence']='ISOLATED_KERNEL_MEASUREMENT';qualified.append(item)
        else:
            item['evidence']='DIAGNOSTIC';diagnostic.append(item)
    write(ROOT/'analysis/actual_serving_API_operator_analysis.json',dict(name=name,
        source_SHA256=digest(path),operator_rows=len(rows),qualified_kernel_profiles=qualified,
        diagnostic_profiles=diagnostic,matched_projection_models=comparison,
        kernel_identity=sorted({k for p in profiles for k in p.get('kernels',{})}),
        hardware_traffic='NOT_MEASURED',SM_counters='NOT_MEASURED',
        online_exclusive_GPU_service='NOT_MEASURED',
        scope='Actual batch-invariant serving Linear API. Clean timing samples and profiled samples remain separate. Failed small-operator observer gates do not disqualify other shapes or silently qualify this whole online serving path.'))
    print('ACTUAL_API_PROJECTION_MODELS',[(r['layers'],round(r['Token_over_Layer_D_projection_model'],4)) for r in comparison],
        'ISOLATED_PROFILE_GATES',len(qualified),len(profiles),flush=True)


if __name__=='__main__':
    main()
