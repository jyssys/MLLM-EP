"""Report the small fallback audit without a broad systems or novelty claim."""
import csv
import statistics
import sys
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer


def main():
    sys.path.insert(0,'/home/esjung/external/looped-exact-depth-chunk-assets/cpu_deps')
    from math_verify import parse,verify,LatexExtractionConfig
    gold={r['id']:r['answer'] for r in read(ROOT/'results/frozen_questions.json')}
    records=[]
    ownership=[]
    groups=[]
    raw_by_tp={}
    for tp in [1,2,4]:
        directory=ROOT/f'results/exploratory_B1_TP{tp}_v1'
        assert read(directory/'guard_status.json')['success']
        rows=read(directory/'natural_requests.json')
        assert len(rows)==6
        raw_by_tp[tp]=rows
        for row in rows:
            answer,rule=final_answer(row['text'])
            correct=False
            if answer is not None:
                try:
                    a=parse('$'+gold[row['id']]+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                    b=parse('$'+answer+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                    correct=bool(verify(a,b,strict=True,timeout_seconds=2))
                except Exception:
                    pass
            records.append({k:v for k,v in row.items() if k not in ['prompt_ids','generated_ids','text','metrics']} |
                dict(extracted_answer=answer,parse_rule=rule,answer_matches=correct,
                     correct_completed=correct and row['finish_reason']=='stop'))
        repeat_checks=[]
        for problem in sorted({r['id'] for r in rows}):
            selected=[r for r in rows if r['id']==problem]
            repeat_checks.append(dict(id=problem,token_repeat_equal=all(r['generated_ids']==selected[0]['generated_ids'] for r in selected),
                E2E_ms=[r['clean_call_wall_ms'] for r in selected],output_lengths=[r['generated_tokens'] for r in selected]))
        groups.append(dict(TP=tp,independent_problems=2,technical_repeats_per_problem=3,
            completed=sum(r['finish_reason']=='stop' for r in rows),total=6,
            correct_completed=sum(r['correct_completed'] for r in records if r['TP']==tp),
            median_output_tokens=statistics.median(r['generated_tokens'] for r in rows),
            median_clean_E2E_ms=statistics.median(r['clean_call_wall_ms'] for r in rows),
            median_backend_TTFT_ms=statistics.median(r['backend_TTFT_ms'] for r in rows),
            median_average_cached_TPOT_ms=statistics.median(r['average_cached_TPOT_ms'] for r in rows),
            repeats=repeat_checks))
        for rank in read(directory/'ownership.json'):
            ownership.append(dict(TP=tp,rank=rank['rank'],GPU_uuid=rank['GPU_uuid'],
                local_KV_heads=rank['geometry'][0]['KV_heads'],cache_slots=rank['cache_slots'],
                weight_unique_bytes=rank['unique_parameter_storage_bytes'],KV_pool_bytes=rank['unique_KV_pool_storage_bytes'],
                KV_pool_blocks=rank['KV_pool_blocks'],memory_allocated=rank['memory_allocated'],
                memory_reserved=rank['memory_reserved']))
    output=ROOT/'results/exploratory_B1_summary.json'
    assert not output.exists()
    cross_tp=[]
    for anchor in raw_by_tp[1]:
        for tp in [2,4]:
            other=next(row for row in raw_by_tp[tp]
                if (row['id'],row['repeat'])==(anchor['id'],anchor['repeat']))
            left,right=anchor['generated_ids'],other['generated_ids']
            cross_tp.append(dict(id=anchor['id'],repeat=anchor['repeat'],TP=tp,
                TP1_output_tokens=len(left),other_output_tokens=len(right),exact_tokens_equal=left==right,
                first_divergent_token_index=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),None),
                TP1_finish=anchor['finish_reason'],other_finish=other['finish_reason']))
    write(output,dict(status='PARTIAL_PROFILE_ONLY',groups=groups,rank_memory=ownership,
        cross_TP_natural_output_comparisons=cross_tp,
        quality_scope='Two continuity MATH problems, greedy. Not a 30-problem AIME or MATH100 score.',
        uncertainty='Only two independent problems. Technical repeats do not create more independent tasks.',
        work_shape='Natural outputs can differ across TP. E2E ratios alone do not establish a matched-shape hardware speedup.',
        not_run=['Simultaneous TPxDP','Concurrency sweep','Token-index ITL','GPU critical-path decomposition'],
        no_loop_specific_claim=True))
    with (ROOT/'results/exploratory_B1_requests.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,field,label in [(axes[0],'clean_call_wall_ms','Clean call wall, ms'),
                              (axes[1],'average_cached_TPOT_ms','Average cached TPOT, ms')]:
        for index,problem in enumerate(sorted({r['id'] for r in records})):
            values=[[r[field] for r in records if r['TP']==tp and r['id']==problem] for tp in [1,2,4]]
            axis.errorbar([1,2,4],[np.mean(v) for v in values],
                yerr=[[np.mean(v)-min(v) for v in values],[max(v)-np.mean(v) for v in values]],
                marker='o',label=f'Continuity problem {index+1}')
        axis.set_xlabel('TP degree');axis.set_xticks([1,2,4]);axis.set_ylabel(label);axis.legend(fontsize=8)
    fig.suptitle('Exploratory only: two natural MATH problems, three technical repeats')
    fig.tight_layout();fig.savefig(ROOT/'figures/exploratory_b1_natural.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=[f"TP{row['TP']} r{row['rank']}" for row in ownership]
    for axis,field,title in [(axes[0],'weight_unique_bytes','Unique loaded weight storage'),
                             (axes[1],'KV_pool_bytes','Reserved KV pool, not filled request KV')]:
        axis.bar(labels,[row[field]/2**30 for row in ownership])
        axis.set_title(title)
        axis.set_ylabel('GiB per rank')
        axis.tick_params(axis='x',rotation=40)
    fig.tight_layout();fig.savefig(ROOT/'figures/exploratory_rank_memory.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
