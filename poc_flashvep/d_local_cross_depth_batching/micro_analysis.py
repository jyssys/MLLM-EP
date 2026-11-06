"""Separate generic batching gains from depth-label and readiness effects."""
import statistics
from .common import *
from .analyze import pct,table
from .qualification import measured


def eligible_cells(folder):
    path=folder/'full_core_micro.json'
    if not path.exists():
        return []
    rows=json.loads(path.read_text())
    # Version 3 initially exhausted its case quota on two 2-token stragglers.
    # C128/B2 contained extra prompt work. Use the strict replacement instead.
    if folder.name.endswith('fair_grid_s1'):
        rows=[r for r in rows if r['chunk'] in (16,32,64) or (r['chunk']==128 and r['decode_batch']==1)]
    return [r for r in rows if r.get('control_version',0)>=3]


def output_audit(model,folder):
    reference=measured(ROOT/'raw'/f'local_{model}_native_n8_s2026110701_off_state')
    completed={}
    summary=folder/'micro_summary.json'
    if summary.exists():
        completed=json.loads(summary.read_text())['completed']
    rows=[]
    cfg=json.loads((ROOT/'configs'/(folder.name+'.json')).read_text())
    for rid,value in completed.items():
        case=int(rid.split('_')[0][4:])
        spec=cfg.get('micro_grid',[dict(batch=cfg.get('micro_decode_batch',2))])[case]
        index=int(rid.rsplit('_',1)[1]) if '_decode_' in rid else spec['batch']
        expected=reference[index]
        rows.append(dict(request_id=rid,prompt_index=index,output_tokens=len(value['tokens']),
            output_exact=value['tokens']==expected['output_ids'][:len(value['tokens'])],
            fixed_depth_exact=value['depths']==expected['exit_depths'][:len(value['depths'])]))
    opportunity=folder/'opportunity_micro.json'
    if opportunity.exists():
        for cohort in json.loads(opportunity.read_text()):
            for rid,value in cohort['completed'].items():
                index=2 if rid=='local' else int(rid.rsplit('_',1)[1])
                expected=reference[index]
                rows.append(dict(method=cohort['method'],repetition=cohort['repetition'],request_id=rid,
                    output_tokens=len(value['tokens']),output_exact=value['tokens']==expected['output_ids'][:len(value['tokens'])],
                    fixed_depth_exact=value['depths']==expected['exit_depths'][:len(value['depths'])]))
    result=dict(passed=bool(rows) and all(r['output_exact'] and r['fixed_depth_exact'] for r in rows),
        streams=len(rows),requests=rows,scope='Generated prefix matches native runtime. This does not replace N32 whole D-local KV/hidden qualification.')
    write(folder/'micro_output_parity.json',result)
    return result


def main():
    core=[];dense=[];dense_checks=[];opportunities=[];verdicts={};profiles=[]
    for model in ('ouro','huginn'):
        for name in (f'micro_{model}_fair_grid_s1',f'micro_{model}_matched_c128_v4',f'micro_{model}_matched_c248_v1',f'micro_{model}_strict_grid_v5'):
            folder=ROOT/'raw'/name
            if not (folder/'status.json').exists():
                continue
            output_audit(model,folder)
            for item in eligible_cells(folder):
                row=dict(model=model,name=name,chunk=item['chunk'],decode_batch=item['decode_batch'],
                    depth_class=item['depth_class'],depths=item['depths'],positions=item['positions'],
                    hidden_exact=item['hidden_exact'],KV_exact=item['KV_exact'],
                    control_version=item.get('control_version',0),primary_timing_eligible=item.get('control_version',0)>=5,
                    warm_repetitions=len(item.get('paired_repetitions',[])),independent_starts=1,
                    graph_eager_fallbacks=item.get('graph_eager_fallbacks'),warmup_host_s=item.get('warmup_host_s'))
                if not item['hidden_exact'] or not item['KV_exact']:
                    row['status']='BLOCKED_EXACTNESS_NO_TIMING';core.append(row);continue
                for method in ('batched','separate'):
                    samples=[x for rep in item['paired_repetitions'] for x in rep[method+'_ms']]
                    for q in (50,95,99):row[f'{method}_p{q}_ms']=pct(samples,q)
                    row[method+'_max_ms']=max(samples)
                row['batch_over_separate']=row['batched_p50_ms']/row['separate_p50_ms']
                core.append(row)
        path=ROOT/'raw'/f'micro_{model}_kernel_profile_v5/core_kernel_profiles.json'
        if path.exists():
            for method,value in json.loads(path.read_text()).items():
                profiles.append(dict(model=model,method=method,chunk=32,decode_batch=2,
                    kernel_count=value['kernels'],profiled_kernel_sum_ms=value['profiled_kernel_sum_ms'],
                    named_attention_kernel_sum_ms=sum(v['duration_us'] for v in value['attention_kernels'])/1000,
                    named_dense_kernel_sum_ms=sum(v['duration_us'] for v in value['dense_kernels'])/1000,
                    actual_weight_read_bytes='NOT_MEASURED',
                    scope=value['scope']+' Name-based categories can be incomplete. Kernel sums are not exclusive wall time.'))
        name=f'micro_{model}_strict_grid_v5' if (ROOT/'raw'/f'micro_{model}_strict_grid_v5/status.json').exists() else f'micro_{model}_fair_grid_s1'
        for item in json.loads((ROOT/'raw'/name/'dense_micro.json').read_text()):
            if item.get('control_version',0)<3:continue
            dense_checks.append(dict(model=model,name=item['name'],exact=item['exact'],status=item.get('status','QUALIFIED')))
            if not item['exact']:continue
            row=dict(model=model,name=item['name'],rows=item['rows'],columns=item['columns'],exact=item['exact'],
                same_cross_identical_graph=item['identical_same_cross_graph'])
            for method in ('separate','same_depth','cross_depth'):
                values=[v for repeat in item['paired_repetitions'] for v in repeat[method+'_ms']]
                for q in (50,95,99):row[f'{method}_p{q}_ms']=pct(values,q)
                row[method+'_max_ms']=max(values)
            row['cross_over_same']=row['cross_depth_p50_ms']/row['same_depth_p50_ms']
            dense.append(row)
        name='opportunity_ouro_graph_control_v2' if model=='ouro' else 'opportunity_huginn_s1'
        folder=ROOT/'raw'/name
        if not (folder/'status.json').exists():continue
        audit=output_audit(model,folder)
        values=json.loads((folder/'opportunity_micro.json').read_text())
        times={}
        for method in ('separate','same','cross'):
            selected=[v for v in values if not v['warmup'] and v['method']==method]
            times[method]=statistics.median(v['cohort_host_ms'] for v in selected)
            replay_clean=all(all(v['graph_after'][key]['captures']==v['graph_before'][key]['captures']
                and v['graph_after'][key]['fallbacks']==v['graph_before'][key]['fallbacks'] for key in v['graph_before']) for v in selected)
            opportunities.append(dict(model=model,method=method,cohort_host_ms=times[method],
                local_first_ms=statistics.median(v['local_first_ms'] for v in selected),
                warm_repetitions=len(selected),independent_starts=1,
                min_cohort_ms=min(v['cohort_host_ms'] for v in selected),max_cohort_ms=max(v['cohort_host_ms'] for v in selected),
                no_new_warm_capture_or_fallback=replay_clean,output_exact=audit['passed'],
                scope='Closed-loop one local prompt with two decode streams. Not 1P1D online serving.'))
        local_times={m:next(v['local_first_ms'] for v in opportunities if v['model']==model and v['method']==m)
            for m in ('separate','same','cross')}
        verdicts[model]=dict(cross_over_same=times['cross']/times['same'],cross_over_separate=times['cross']/times['separate'],
            local_prompt_first_token_cross_over_same=local_times['cross']/local_times['same'],
            incremental_full_slice_saving_fraction=1-local_times['cross']/local_times['same'],
            incremental_dense_gain=False,dense_label_control='Same and cross use the identical graph and input tensor.',
            Phase_C_gate_pass=False,Phase_D='NOT_RUN_PHASE_C_GATE',
            scope='No incremental dense mechanism demonstrated. Do not reinterpret generic batching savings as depth-specific gain.')
        independent=[]
        for start in ((3,4,5) if model=='ouro' else (2,3,4)):
            name=f'opportunity_{model}_graph_start{start}';path=ROOT/'raw'/name/'opportunity_micro.json'
            if not (path.parent/'status.json').exists():continue
            audit=output_audit(model,path.parent);data=json.loads(path.read_text())
            methods={m:[v for v in data if not v['warmup'] and v['method']==m] for m in ('separate','same','cross')}
            row=dict(model=model,start=start,name=name,output_exact=audit['passed'])
            for method,items in methods.items():
                row[method+'_cohort_ms']=statistics.median(v['cohort_host_ms'] for v in items)
                row[method+'_local_first_ms']=statistics.median(v['local_first_ms'] for v in items)
                row[method+'_cross_phase_submissions']=statistics.median(sum(s['count'] for s in v['submissions'] if s['cross_phase']) for v in items)
                depth_observed=all('cross_depth' in s for v in items for s in v['submissions'])
                row[method+'_cross_depth_submissions']=statistics.median(sum(s['count'] for s in v['submissions'] if s['cross_depth']) for v in items) if depth_observed else None
                row[method+'_cross_phase_cross_depth_submissions']=statistics.median(sum(s['count'] for s in v['submissions'] if s['cross_depth'] and s['cross_phase']) for v in items) if depth_observed else None
                tails=[];tpots=[]
                for v in items:
                    for rid,events in v['token_events'].items():
                        if rid=='local':continue
                        fresh=[];seen_count=0
                        for stamp,count in events:
                            if count>seen_count:fresh.append(stamp);seen_count=count
                        if len(fresh)>1:
                            tails.extend(1000*(b-a) for a,b in zip(fresh,fresh[1:]));tpots.append(1000*(fresh[-1]-fresh[0])/(len(fresh)-1))
                row[method+'_TPOT_p95_ms']=pct(tpots,95);row[method+'_ITL_p99_ms']=pct(tails,99)
                row[method+'_max_ITL_ms']=max(tails,default=None)
            row['cohort_cross_over_same']=row['cross_cohort_ms']/row['same_cohort_ms']
            row['local_cross_over_same']=row['cross_local_first_ms']/row['same_local_first_ms']
            independent.append(row)
        if independent:
            table(ROOT/f'{model}_independent_readiness.csv',independent)
            import numpy as np
            rng=np.random.default_rng(2026110608)
            summary={}
            for metric in ('cohort_cross_over_same','local_cross_over_same'):
                ratios=np.array([v[metric] for v in independent])
                samples=np.median(rng.choice(ratios,(10000,len(ratios)),replace=True),axis=1)
                summary[metric]=dict(median=float(np.median(ratios)),starts=len(ratios),
                    bootstrap_95_CI=[float(x) for x in np.percentile(samples,[2.5,97.5])] if len(ratios)>=3 else None,
                    scope='Exploratory paired-start bootstrap of closed-loop controls. Not an online-capacity confidence interval.')
            verdicts[model]['independent_readiness']=summary
        for chunk in (128,248):
            variants=('_v1','_alignment_v1','_d_device_v1') if chunk==248 else ('_v1','_alignment_v1')
            for suffix in variants:
                aligned=suffix!='_v1'
                name=f'opportunity_{model}_c{chunk}'+suffix
                path=ROOT/'raw'/name/'opportunity_micro.json'
                if not (path.parent/'status.json').exists():continue
                audit=output_audit(model,path.parent)
                data=json.loads(path.read_text())
                for method in ('separate','same','same_align','cross'):
                    selected=[v for v in data if not v['warmup'] and v['method']==method]
                    if not selected:continue
                    itls=[];tpots=[]
                    within=[]
                    for v in selected:
                        local_first=v['token_events']['local'][0][0]
                        local_start=local_first-v['local_first_ms']/1000
                        background=[(v['token_events'][rid][0][0],value['end']) for rid,value in v['completed'].items() if rid!='local']
                        from .profile_analysis import union
                        within.append(any(a<=local_start<=local_first<=b for a,b in union(background)))
                        for rid,events in v['token_events'].items():
                            if rid=='local':continue
                            stamps=[];count=0
                            for stamp,num in events:
                                if num>count:stamps.append(stamp);count=num
                            if len(stamps)>1:
                                itls.extend(1000*(b-a) for a,b in zip(stamps,stamps[1:]))
                                tpots.append(1000*(stamps[-1]-stamps[0])/(len(stamps)-1))
                    opportunities.append(dict(model=model,name=name,chunk=chunk,alignment_control=aligned,method=method,
                        physical_GPU=json.loads((ROOT/'configs'/(name+'.json')).read_text())['gpus'][0],
                        cohort_host_ms=statistics.median(v['cohort_host_ms'] for v in selected),
                        local_first_ms=statistics.median(v['local_first_ms'] for v in selected),
                        warm_repetitions=len(selected),independent_starts=1,output_exact=audit['passed'],
                        TPOT_p95_ms=pct(tpots,95),ITL_p99_ms=pct(itls,99),max_ITL_ms=max(itls,default=None),
                        alignment_steps=statistics.median(v.get('one_step_alignment_submissions',0) for v in selected),
                        new_capture_or_fallback_free=all(all(
                            v['graph_after'][key]['captures']==v['graph_before'][key]['captures']
                            and v['graph_after'][key]['fallbacks']==v['graph_before'][key]['fallbacks']
                            for key in v['graph_before']) for v in selected),
                        whole_prefill_within_existing_decode_repetitions=sum(within),
                        mixed_phase_submissions=statistics.median(sum(s['count'] for s in v.get('submissions_after_local_enqueue',[])
                            if s['cross_phase']) for v in selected),
                        mixed_phase_different_depth_submissions=statistics.median(sum(s['count'] for s in v.get('submissions_after_local_enqueue',[])
                            if s['cross_phase'] and s.get('cross_depth',False)) for v in selected),
                        core_groups_after_local=statistics.median(sum(s['count']*s['loops'] for s in v.get('submissions_after_local_enqueue',[])) for v in selected),
                        scope='Exploratory closed-loop larger-chunk control. One independent start. Not online PD serving.'))
    table(ROOT/'matched_core_micro.csv',core);table(ROOT/'dense_micro.csv',dense)
    table(ROOT/'dense_correctness.csv',dense_checks)
    table(ROOT/'micro_kernel_diagnostics.csv',profiles)
    table(ROOT/'readiness_opportunity.csv',opportunities)
    write(ROOT/'phase_c_gate.json',verdicts)
    print(json.dumps(verdicts))


if __name__=='__main__':main()
