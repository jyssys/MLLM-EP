"""Aggregate qualified PP controls. Keep the blocked TP matrix explicit."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contract import ROOT, cases, write, sha


def load(path):
    path=Path(path)
    return json.loads(gzip.open(path,'rt').read() if path.suffix=='.gz' else path.read_text())


def csv_file(name, rows):
    with (ROOT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def metrics(arm, runtime, repeat, index):
    label=f'C0_{arm}_r{repeat}' if runtime=='legacy' else f'PP_transport_clean_{arm}_r{repeat}'
    directory=ROOT/'raw'/label
    if runtime=='legacy':
        r=load(directory/f'case_{index:02d}.json.gz')
        outputs=r['outputs'];start=0.;times=[o['token_times_s'] for o in outputs]
        memory=r['memory'];preparation=r['preparation_wall_s'];graphs=sum(r['graph_captures_during_run'])
        transfers=r['transfer_counts'];bytecounts=r['transfer_bytes']
        assert r['locked_output_agreement']
        wall=r['wall_s']
    else:
        ranks=[load(directory/f'case_{index:02d}_rank{k}.json.gz') for k in range(4)]
        r=ranks[3];outputs=list(r['outputs'].values());start=ranks[0]['start_abs_s']
        times=[[t-start for t in o['token_times_abs_s']] for o in outputs]
        wall=max(t[-1] for t in times)
        memory=[dict(rank=k,physical_GPU=k+4,**v['memory']) for k,v in enumerate(ranks)]
        preparation=max(v['preparation_wall_s'] for v in ranks)
        graphs=sum(v['graph_captures_during_run'] for v in ranks)
        transfers=[v['transfer_counts'] for v in ranks];bytecounts=[v['transfer_bytes'] for v in ranks]
        frozen={v['id']:v['teacher_ids'] for v in cases()[index]['requests']}
        assert all(o['ids']==frozen[o['id']] for o in outputs)
    assert graphs==0
    C=cases()[index]['C'];O=cases()[index]['O']
    ttft=[t[0] for t in times];e2e=[t[-1] for t in times]
    tpot=[(t[-1]-t[0])/(len(t)-1) for t in times]
    hidden_keys=['hidden'] if runtime=='legacy' else ['prefill','decode']
    row=dict(evidence='LIVE_MEASURED_PP_ONLY_CONTROL',arm=arm,runtime=runtime,repeat=repeat,index=index,
        C=C,P=128,O=O,makespan_s=wall,output_tokens_s=C*O/wall,requests_s=C/wall,
        TTFT_p50_ms=float(np.percentile(ttft,50))*1000,TTFT_p90_ms=float(np.percentile(ttft,90))*1000,
        TPOT_p50_ms=float(np.percentile(tpot,50))*1000,TPOT_p90_ms=float(np.percentile(tpot,90))*1000,
        E2E_p50_ms=float(np.percentile(e2e,50))*1000,E2E_p90_ms=float(np.percentile(e2e,90))*1000,
        preparation_s=preparation,graph_captures_in_clock=graphs,
        hidden_peer_receives=sum(sum(v.get(k,0) for k in hidden_keys) for v in transfers),
        hidden_peer_bytes=sum(sum(v.get(k,0) for k in hidden_keys) for v in bytecounts),
        scalar_token_receives=sum(v.get('token',v.get('token_return',0)) for v in transfers),
        source=str(directory.relative_to(ROOT)))
    memory_rows=[]
    for k,m in enumerate(memory):
        weights=m.get('weights',m.get('weights_bytes'))
        memory_rows.append(dict(arm=arm,runtime=runtime,repeat=repeat,index=index,rank=k,physical_GPU=k+4,
            weight_bytes=weights['total'],KV_bytes=m['KV_bytes'],allocated_bytes=m['allocated_bytes'],
            reserved_bytes=m['reserved_bytes'],peak_allocated_bytes=m['peak_bytes'],
            graph_scratch_bytes=m.get('graph_scratch_bytes','NOT_AVAILABLE'),
            IPC_export_bytes=m.get('peer_export_bytes','NOT_APPLICABLE'),
            scope='PyTorch allocator. No sampled NVML serving peak or capacity claim.'))
    return row,memory_rows


def main():
    excluded=[]
    for path in sorted((ROOT/'raw').glob('PP_transport_*')):
        if path.name.startswith('PP_transport_clean_'):continue
        excluded.append(str(path.relative_to(ROOT)))
    for name in ['mp_correctness_layer_pp4','mp_correctness_loop_pp4','tp_operator_diagnostic']:
        excluded.append('profiles/'+name+'.nsys-rep')
    write(ROOT/'excluded_runs.json',dict(
        reason='This task\'s idle native vLLM workers remained resident. These later controls are excluded, not deleted.',
        cleanup='orphan_cleanup_audit.json. The recorded task-owned process group is no longer resident.',excluded=excluded,
        valid_replacements='PP_transport_clean_*, profiles/clean_mp_* and profiles/clean2_tp_*',
        interrupted_profile='profiles/clean_tp_operator_diagnostic.nsys-rep. Driver exit was interrupted after capture. Use clean2 instead.'))
    rows=[];memory=[]
    for runtime in ['legacy','mp']:
        for arm in ['layer_pp4','loop_pp4']:
            for repeat in range(3):
                for index in [0,1]:
                    row,m=metrics(arm,runtime,repeat,index);rows.append(row);memory.extend(m)
    csv_file('pp_controls_raw.csv',rows);csv_file('memory_raw.csv',memory)
    summary=[]
    for runtime in ['legacy','mp']:
        for index in [0,1]:
            for arm in ['layer_pp4','loop_pp4']:
                selected=[r for r in rows if r['runtime']==runtime and r['index']==index and r['arm']==arm]
                out=dict(runtime=runtime,index=index,C=cases()[index]['C'],arm=arm,repeats=3)
                for key in ['makespan_s','output_tokens_s','TTFT_p50_ms','TTFT_p90_ms','TPOT_p50_ms','TPOT_p90_ms','E2E_p50_ms','E2E_p90_ms']:
                    out[key+'_median']=float(np.median([r[key] for r in selected]))
                out['makespan_s_min']=min(r['makespan_s'] for r in selected)
                out['makespan_s_max']=max(r['makespan_s'] for r in selected)
                summary.append(out)
    csv_file('pp_controls_summary.csv',summary)
    rng=np.random.default_rng(20261020);paired=[]
    for runtime in ['legacy','mp']:
        for index in [0,1]:
            values={arm:np.array([next(r['makespan_s'] for r in rows if r['runtime']==runtime and r['arm']==arm and r['index']==index and r['repeat']==rep) for rep in range(3)]) for arm in ['layer_pp4','loop_pp4']}
            draws=rng.integers(0,3,size=(5000,3))
            ratios=np.median(values['layer_pp4'][draws],axis=1)/np.median(values['loop_pp4'][draws],axis=1)
            paired.append(dict(runtime=runtime,index=index,C=cases()[index]['C'],
                layer_over_loop_speedup=float(np.median(values['layer_pp4'])/np.median(values['loop_pp4'])),
                paired_cohort_bootstrap_95CI=[float(x) for x in np.percentile(ratios,[2.5,97.5])],
                bootstrap_draws=5000,sampling_unit='One paired physical cohort repeat',
                limitation='Only three pairs. This interval does not establish population significance.'))
    write(ROOT/'pp_controls_paired.json',paired)
    matrix=[]
    for c in cases():
        matrix.append(dict(index=c['index'],R=4,C=c['C'],P=c['P'],O=c['O'],
            LPP4_legacy='PP_ONLY_CONTROL' if c['index'] in [0,1] else 'NOT_RUN',
            LoopPP4_legacy='PP_ONLY_CONTROL' if c['index'] in [0,1] else 'NOT_RUN',
            LPP4_mp='PP_ONLY_CONTROL' if c['index'] in [0,1] else 'NOT_RUN',
            LoopPP4_mp='PP_ONLY_CONTROL' if c['index'] in [0,1] else 'NOT_RUN',
            TP4_matched='NOT_RUN_C2_NUMERICAL_GATE',TP4_vLLM_serving='NOT_RUN_C2_NUMERICAL_GATE'))
    csv_file('primary_matrix_status.csv',matrix)
    numeric=[]
    for file in sorted((ROOT/'correctness').glob('*rank0.json')):
        values=load(file)
        if not isinstance(values,list):continue
        for v in values:
            if not v.get('teacher_forced'):continue
            numeric.append(dict(source=file.name,P=v['P'],O=v['O'],id=v['id'],PASS=v['PASS'],
                max_logit_abs=max(x['max_abs'] for x in v['logit_errors']),
                max_logit_mean_abs=max(x['mean_abs'] for x in v['logit_errors']),
                max_hidden_relative_L2=max(x['relative_L2'] for x in v['hidden_errors']),
                KV_relative_L2=v['KV_relative_L2'],all_aligned_argmax_equal=v['all_token_agreement']))
    csv_file('tp_numerical_diagnostics.csv',numeric)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,index in zip(axes,[0,1]):
        for offset,arm in [(-.18,'layer_pp4'),(.18,'loop_pp4')]:
            selected=[next(r for r in summary if r['runtime']==runtime and r['index']==index and r['arm']==arm) for runtime in ['legacy','mp']]
            centers=np.arange(2)+offset
            med=[r['makespan_s_median'] for r in selected]
            ax.bar(centers,med,.35,label=arm)
            ax.errorbar(centers,med,yerr=[[m-r['makespan_s_min'] for m,r in zip(med,selected)],[r['makespan_s_max']-m for m,r in zip(med,selected)]],fmt='none',color='black',capsize=3)
        ax.set(xticks=[0,1],xticklabels=['Legacy','Four-process'],title=f'PP-only control: C{cases()[index]["C"]}, P128/O32',ylabel='Cohort makespan (s)');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'pp_controls.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ext=load(ROOT/'external_numerical_comparison.json');ext=[r for r in ext if r['rank']==0]
    ax.plot([r['P'] for r in ext],[r['max_hidden_relative_L2'] for r in ext],marker='o',label='Native vLLM TP4: numerical observer')
    ax.axhline(.03,color='red',linestyle='--',label='Frozen hidden relative L2 limit')
    ax.set(xlabel='Prompt tokens',ylabel='Maximum hidden relative L2 across aligned steps',title='Token agreement does not imply hidden-state equivalence');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'numerical_gate.png',dpi=160);plt.close(fig)
    write(ROOT/'result_summary.json',dict(verdict='BLOCKED',blocking_gate='C2_FULL_MODEL_NUMERICAL_EQUIVALENCE',
        architectural_conclusion='UNRESOLVED',TP_serving='NOT_RUN',PP_controls=paired,
        report='analysis.md',frozen_spec_sha256=sha(ROOT/'spec_frozen.md')))
    print('AGGREGATED',len(rows),'PP-only cohort measurements. TP serving: NOT_RUN.',flush=True)


if __name__=='__main__':main()
