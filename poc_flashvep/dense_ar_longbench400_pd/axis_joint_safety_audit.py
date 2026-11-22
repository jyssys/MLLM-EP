"""Report the stricter subset where both axes satisfy the Native decode gates."""
import collections
import json
import time
from .common import ROOT, REPO, write, digest


def main():
    path=ROOT/'analysis/axis_matrix_heldout_confirmation_r2.json'
    data=json.loads(path.read_text());rows=[]
    for cell in data['cells']:
        tests=cell.get('native_relative',{})
        safe={a:bool(tests.get(a,{}).get('decode_5percent_pass'))
              and bool(tests.get(a,{}).get('absolute_decode_pass')) for a in ('Layer','Token')}
        both=all(safe.values())
        winner=cell.get('latency_direction') if cell['status'].endswith('_CONFIRMED') else None
        rows.append(dict(input_bucket=cell['input_bucket'],output=cell['output'],cut=cell['cut'],QPS=cell['QPS'],
            original_status=cell['status'],safe=safe,both_axes_decode_safe=both,
            both_safe_confirmed_winner=winner if both else None,
            winner_also_beats_native_TTFT=bool(winner and both and tests[winner]['native_TTFT_gain']),
            Layer_over_Token_TTFT=cell.get('TTFT_p95_Layer_over_Token')))
    assert len(rows)==64
    subset=[r for r in rows if r['both_axes_decode_safe']]
    counts=dict(collections.Counter(r['original_status'] for r in subset))
    output=dict(time=time.time(),rows=rows,both_safe_cells=len(subset),both_safe_counts=counts,
        both_safe_native_TTFT_winner_counts=dict(collections.Counter(r['both_safe_confirmed_winner']
            for r in rows if r['winner_also_beats_native_TTFT'])),
        original_matrix_SHA256=digest(path),evidence='TRACE_ANALYSIS_OF_MEASURED_ONLINE_C0',
        scope='Keep frozen original winner-only safety labels. This supplementary subset requires both arms to pass '
              'Native-relative ITLp99/TPOTp95 CIupper<=1.05 and every-start request decode attainment>=.95. '
              'No new SLO,no dropped raw or relabelled original winner. Pointwise empirical CIs remain unadjusted. '
              'This is C0 actual1/1/1,not natural C3 or sustainable serving capacity.')
    write(ROOT/'analysis/axis64_both_axes_decode_safety_audit_v1.json',output)
    report=REPO/'poc_flashvep/reports/dense_ar_longbench400_input_output_axis_winner_map.md'
    marker='## 양축 모두 Decode Gate를 통과한 더 엄격한 부분집합'
    prefix=report.read_text().split(marker)[0].rstrip()
    lines=['',marker,'',
        '원래 Winner Label은 이긴 Arm의 Decode Gate를 요구한다.',
        '상대 Arm의 실패도 원래 표와 NI Heatmap에 남긴다.',
        '아래 추가 감사는 Layer와 Token이 모두 같은 Native Decode Gate를 통과할 것을 요구한다.',
        '원래 Label과 SLO는 변경하지 않는다. 더 엄격한 해석 범위를 따로 표시한다.','',
        '| 두 Arm 모두 안전한 원래 판정 | Cell 수 |','|---|---:|']
    for label,count in sorted(counts.items()):
        lines.append(f'| {label} | {count} |')
    lines += ['',f'양축 모두 안전한 Cell은 {len(subset)}/64개다.',
        '서로 다른 Cell의 Layer Winner와 Token Winner를 실제 Continuous Serving의 보편 법칙으로 확대하지 않는다.',
        '축 사이 TTFT Win과 Native 대비 TTFT Win은 여전히 다른 판정이다.','',
        '| 양축 모두 안전하지만 Native도 이긴 TTFT Winner | Cell 수 |','|---|---:|']
    for label,count in sorted(output['both_safe_native_TTFT_winner_counts'].items()):
        lines.append(f'| {label} | {count} |')
    lines += ['', 'Native TTFT Win도 해당 Cell의 Paired CI 상한<1을 요구한다.',
        'Goodput 우위는 별도 CI다. TTFT만으로 Goodput Win을 선언하지 않는다.',
        '64개 Cell 전체의 Familywise CI는 아니다. 전체 실패 Arm도 보존했다.','',
        '- [모든 Cell의 양축 Safety와 원래 Label](dense_ar_longbench400_2d_20261121/analysis/axis64_both_axes_decode_safety_audit_v1.json)','']
    report.write_text(prefix+'\n'+'\n'.join(lines),encoding='utf-8')
    figure(output)
    print('BOTH_AXIS_DECODE_SAFETY',len(subset),'/64',counts,flush=True)


def figure(data):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from .figures import save
    colors=['#a9cce3','#f8c899','#e5e7e9','#b7b7b7']
    categories={'LAYER_CONFIRMED':0,'TOKEN_CONFIRMED':1,'TIE_OR_INCONCLUSIVE':2}
    captions=['3K–6K','6K–10K','10K–20K','20K–36K']
    cuts=(2,4,8,12);outputs=(8,16,64,128)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for bucket,ax in enumerate(axes.flat):
        selected=[r for r in data['rows'] if r['input_bucket']==bucket]
        matrix=np.empty((4,4))
        for row in selected:
            i,j=outputs.index(row['output']),cuts.index(row['cut'])
            matrix[i,j]=categories[row['original_status']] if row['both_axes_decode_safe'] else 3
        ax.imshow(matrix,cmap=ListedColormap(colors),vmin=-.5,vmax=3.5,aspect='auto')
        for row in selected:
            i,j=outputs.index(row['output']),cuts.index(row['cut'])
            label=row['both_safe_confirmed_winner'] or ('Tie' if row['both_axes_decode_safe'] else 'Unsafe arm')
            value=row['Layer_over_Token_TTFT']['estimate']
            ax.text(j,i,f'{label}\n{value:.3f}',ha='center',va='center',fontsize=10)
        ax.set_xticks(range(4),['Last'+str(c) for c in cuts])
        ax.set_yticks(range(4),[str(o) for o in outputs])
        ax.set(xlabel='Matched-work cut',ylabel='Controlled output tokens',
            title=f'Input {captions[bucket]} | QPS {selected[0]["QPS"]:.6f}')
    counts=data['both_safe_counts']
    fig.suptitle('C0 axis comparison | BOTH arms must protect Native decode\n'
        f'{counts.get("LAYER_CONFIRMED",0)} Layer / {counts.get("TOKEN_CONFIRMED",0)} Token / '
        f'{counts.get("TIE_OR_INCONCLUSIVE",0)} tie / {len(data["rows"])-data["both_safe_cells"]} not jointly qualified',fontsize=14)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,
        ('Layer CI win','Token CI win','CI includes1','At least one arm fails safety'))],
        loc='lower center',bbox_to_anchor=(.5,.14),ncol=4,fontsize=8)
    caption=('TRACE AUDIT OF MEASURED ONLINE C0 | Qwen3-8B,2xH100,BI0 BF16,B8192/S1024,KV42GiB/block128.\n'
        'Actual batch1/1/1,credit4 | N16/start; n3,three predeclared core cells n5.\n'
        'Cell numbers:Layer/Token TTFTp95 ratio. Labels require both Native ITLp99/TPOTp95 CIupper<=1.05 and every-start decode attainment>=.95.\n'
        'Pointwise unadjusted start-level95% CIs. Unsafe cells retain measured ratios. Axis win is not a Native TTFT or goodput win. Not natural C3.')
    fig.tight_layout(rect=(0,.21,1,.93))
    save(fig,'axis400_both_axes_decode_safe_winner_map',caption,
        dict(evidence=data['evidence'],rows=data['rows'],both_safe_counts=data['both_safe_counts'],
            original_matrix_SHA256=data['original_matrix_SHA256'],scope=data['scope']),manual_layout=True)


if __name__=='__main__':
    main()
