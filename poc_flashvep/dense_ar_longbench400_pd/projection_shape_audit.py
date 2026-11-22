"""Recover actual projection shapes from observer-qualified isolated captures."""
import collections
import csv
import json
import time
import numpy as np
from .common import ROOT, REPO, write, digest
from .profile_analysis import union
from .c0_queue_model import part


PHASES={'P_PREFIX_PREFILL','PREFIX_UPPER_CONTINUATION','SUFFIX_FULL36_CONTINUATION'}


def category(op):
    kind=op['category']
    if kind=='GEMM_O_OR_DOWN':
        return 'GEMM_O_PROJ' if op['module_prefix'].endswith('.o_proj') else 'GEMM_DOWN'
    return kind


def one(name,axis,role,start):
    path=ROOT/'analysis'/(name+'_'+role+'_kernel_decomposition.json')
    data=json.loads(path.read_text())
    cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
    prompts=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    ids={r['step_id'] for r in data['frames'] if r['phase'] in PHASES}
    ops={r['external_id']:r for r in data['actual_operators']
         if r['step_id'] in ids and r['actual_GEMM_MNK'] and r['category']!='GEMM_LM_HEAD'}
    assert data['mapped_kernel_fraction']==1 and ops
    folder=ROOT/'raw'/name/('activity_'+role)
    external={int(r['correlation']):int(r['external_id'])
        for r in csv.DictReader((folder/'kernels.csv.external.csv').open()) if int(r['external_kind'])==3}
    names={int(r['name_id']):r['kernel_name'] for r in csv.DictReader((folder/'kernels.csv.names.csv').open())}
    intervals=collections.defaultdict(list);kernel_names=collections.defaultdict(collections.Counter)
    for row in csv.DictReader((folder/'kernels.csv').open()):
        if row['kind']!='kernel':
            continue
        op=ops.get(external.get(int(row['correlation'])))
        if op is None:
            continue
        key=category(op)
        intervals[key].append((int(row['start_ns']),int(row['end_ns'])))
        kernel_names[key][names[int(row['name_id'])]]+=1
    logical=sum(getattr(part(cfg,i,r['input_tokens']),'P_rows' if role=='prefill' else 'D_rows')
                for i,r in enumerate(prompts))
    categories={}
    for key in sorted({category(op) for op in ops.values()}):
        selected=[r for r in ops.values() if category(r)==key]
        shapes=collections.Counter(tuple(r['actual_GEMM_MNK'][k] for k in ('M','N','K')) for r in selected)
        rows=sum(m*c for (m,n,k),c in shapes.items())
        seconds=union(intervals[key])/1e9
        flops=sum(2*m*n*k*c for (m,n,k),c in shapes.items())
        categories[key]=dict(API_calls=len(selected),GPU_kernel_launches=sum(kernel_names[key].values()),
            physical_M_sum=rows,logical_token_layer_rows=logical,exact_row_match=rows==logical,
            shapes=[dict(M=m,N=n,K=k,API_calls=c) for (m,n,k),c in sorted(shapes.items())],
            projection_GPU_kernel_union_ms_per_request=seconds*1000/cfg['count'],
            dense_linear_FLOPs=flops,effective_dense_linear_TF_per_second=flops/seconds/1e12,
            actual_kernel_names=dict(kernel_names[key]))
    assert set(categories)=={'GEMM_QKV','GEMM_O_PROJ','GEMM_GATE_UP','GEMM_DOWN'}
    overall_seconds=union([v for values in intervals.values() for v in values])/1e9
    return dict(name=name,axis=axis,role=role,start=start,requests=cfg['count'],categories=categories,
        total_projection_GPU_union_ms_per_request=overall_seconds*1000/cfg['count'],
        total_projection_GPU_kernel_launches_per_request=sum(sum(v.values()) for v in kernel_names.values())/cfg['count'],
        source_capsule_SHA256=cfg['source_capsule_SHA256'],dataset_SHA256=cfg['dataset_SHA256'],
        source_decomposition_SHA256=digest(path))


def main():
    parent=ROOT/'analysis/operator_repeat_observer_analysis_v3.json'
    data=json.loads(parent.read_text());rows=[]
    for cell in data['cells']:
        if cell['input_bucket']!=3 or cell['cut']!=4:
            continue
        assert cell['observer_gate_passed'] and len(cell['triplets'])==3
        for t in cell['triplets']:
            for role in ('prefill','decode'):
                rows.append(one(t['jobs'][1],cell['axis'],role,t['start']))
    assert len(rows)==12
    write(ROOT/'analysis/qualified_projection_shape_and_launch_audit_v1.json',dict(time=time.time(),rows=rows,
        observer_audit_SHA256=digest(parent),all_projection_rows_match_logical=all(
            c['exact_row_match'] for r in rows for c in r['categories'].values()),
        evidence='ISOLATED_KERNEL_MEASUREMENT_AND_STRUCTURAL_LINEAR_FLOPS',
        scope='Only the two qualified long-input Last4 cells,three warm-all-shape CPC triplets each. '
              'Projection inputs and weights are actual serving API tensors. Kernel counts and union use CUPTI correlation. '
              'LM head,attention,norm,residual,transfer and decode frames are excluded. API calls are not GPU kernel launches. '
              '2MNK is structural dense linear FLOPs. Dividing it by attributed kernel union gives an effective rate,not SM/Tensor Core utilization. '
              'Do not sum overlapping category times as exclusive total or transfer these isolated rates to natural C3 serving.'))
    report=REPO/'poc_flashvep/reports/dense_ar_longbench400_axis_cost_model.md'
    marker='## 적격 Capture의 실제 Projection Shape와 Launch'
    prefix=report.read_text().split(marker)[0].rstrip()
    lines=['',marker,'',
        '긴 Input/Last4의 적격 Capture만 사용했다. Axis별 독립 CPC Triplet3개다.',
        '각 Profile은 실제 원문16개다. Shape Warmup에는 해당 Prompt의 사전 정보가 사용됐다.',
        '아래 Call과 Kernel 수는 Start당 합계의 평균이다. 시간은 Request당 GPU Union의 평균이다.',
        'API Call 수와 GPU Kernel Launch 수를 구분한다.',
        'Shape는 실제 Serving API의 Input과 Weight에서 읽었다. 구조적 모델 차원만으로 추정하지 않았다.','',
        '| Axis / Role | Projection | 실제 M 범위 | 실제 N/K | API Calls/Start | GPU Kernels/Start | GPU Union(ms/request) | 유효 Dense TFLOP/s |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for axis in ('Layer','Token'):
        for role in ('prefill','decode'):
            selected=[r for r in rows if r['axis']==axis and r['role']==role]
            for key in ('GEMM_QKV','GEMM_O_PROJ','GEMM_GATE_UP','GEMM_DOWN'):
                values=[r['categories'][key] for r in selected]
                shapes=[s for v in values for s in v['shapes']]
                mm=[s['M'] for s in shapes];nk=sorted({(s['N'],s['K']) for s in shapes})
                mean=lambda field:float(np.mean([v[field] for v in values]))
                lines.append(f'| {axis}/{role} | {key} | {min(mm)}–{max(mm)} | {nk} | '
                    f'{mean("API_calls"):.3f} | {mean("GPU_kernel_launches"):.3f} | '
                    f'{mean("projection_GPU_kernel_union_ms_per_request"):.6f} | '
                    f'{mean("effective_dense_linear_TF_per_second"):.3f} |')
    lines += ['', 'TFLOP/s는 구조적 Dense Linear FLOPs `2MNK`를 관측 Kernel Union으로 나눈 유효 Rate다.',
        'SM 활용률, Tensor Core 활용률 또는 HBM Traffic 실측값이 아니다.',
        'Attention, Norm, Residual, LM Head와 Transport 비용은 포함하지 않는다.',
        'Category가 다른 GPU 구간은 겹칠 수 있다. 표의 시간을 Exclusive Total로 합산하지 않는다.',
        '모든 Projection의 실제 M 합과 Logical Token-Layer Rows 일치 여부도 JSON에 보존했다.',
        '기본 B8192 때문에 긴 Layer Query도 여러 Model Call로 나뉜다.',
        '전체 Input Length를 각 GEMM의 M 값으로 대입하지 않는다.',
        '이 결과는 고립된 C0 경로다. 강한 Compiled Stock Token의 Natural C3 Operator 비용을 대신하지 않는다.','',
        '- [실제 모든 Shape, Kernel 이름과 Raw Hash](dense_ar_longbench400_2d_20261121/analysis/qualified_projection_shape_and_launch_audit_v1.json)','']
    report.write_text(prefix+'\n'+'\n'.join(lines),encoding='utf-8')
    figures(rows)
    print('QUALIFIED_PROJECTION_SHAPE_AUDIT',len(rows),'profile-role records',flush=True)


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .confirmed_figures import mean_ci
    from .figures import save
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    points=[]
    for index,role in enumerate(('prefill','decode')):
        for column,(field,label) in enumerate((
            ('total_projection_GPU_kernel_launches_per_request','Projection kernel launches/request'),
            ('total_projection_GPU_union_ms_per_request','Projection GPU kernel union (ms/request)'))):
            ax=axes[index,column]
            for xpos,axis,color in ((0,'Layer','#0072b2'),(1,'Token','#cc79a7')):
                values=[r[field] for r in rows if r['axis']==axis and r['role']==role]
                mean,ci=mean_ci(values)
                ax.bar(xpos,mean,color=color,alpha=.8)
                ax.errorbar(xpos,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='none',ecolor='black',capsize=3)
                ax.annotate(f'{mean:.3f}',(xpos,mean),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
                points.append(dict(axis=axis,role=role,field=field,mean=mean,CI95=ci,independent_profile_starts=3))
            ax.set(xticks=[0,1],xticklabels=['Layer Last4','Token matched rows'],ylabel=label,
                title=('P prefix' if role=='prefill' else 'D continuation')+' | Same observed projection rows')
            ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Qualified long-input serving-API calibration | Launch count and GEMM cost are different')
    save(fig,'qualified_projection_shape_efficiency',
        'ISOLATED KERNEL MEASUREMENT | N16/profile | Three independent warm-all-shape CPC triplets per axis | Both roles pass5% observer gate.\n'
        'Input20006–31133,Last4,BI0,B8192/S1024. Observed D M:Layer1036–8192,Token2224–3456. Projection excludes LM head and attention.\n'
        'Error bars:pointwise profile-start mean empirical95% CI,27 resamples. Actual GPU unions,not host enclosures.\n'
        'Warmup uses prompt-shape foreknowledge. Not short-input evidence,natural C3 operator fractions,SM/HBM utilization,or online serving speedup.',
        dict(evidence='ISOLATED_KERNEL_MEASUREMENT',points=points,
             audit='qualified_projection_shape_and_launch_audit_v1.json'))


if __name__=='__main__':
    main()
