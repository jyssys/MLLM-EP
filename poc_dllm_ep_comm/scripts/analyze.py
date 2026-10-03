"""Request bootstrap and strict accounting; no latency projections or method."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import pandas as pd
from poc_dllm_ep_comm.contract import ROOT,write
from poc_dllm_ep_comm.metrics import traffic,bootstrap_mean,graph_stats,decision,MASK_ID,EOS_ID

def read(p):return json.loads(p.read_text())

def request(directory):
    traces=[read(directory/f'trace.rank{r}.json') for r in range(4)]
    arrays=[np.load(directory/f'trace.rank{r}.npz') for r in range(4)]
    results={mode:[read(directory/f'{mode}.result.rank{r}.json') for r in range(4)] for mode in ('bypass','timing','trace')}
    for mode,values in results.items():
        assert all(v['output_ids']==values[0]['output_ids'] and v['nfe']==values[0]['nfe'] for v in values),mode+' cross-rank parity'
    parity=all(results[m][0]['output_ids']==results['bypass'][0]['output_ids'] and
        results[m][0]['nfe']==results['bypass'][0]['nfe'] for m in ('timing','trace'))
    assert parity,'instrumentation changed outputs or diffusion schedule'
    if (directory/'acceptance.rank0.json').exists():
        a=read(directory/'acceptance.rank0.json')
        assert a['final_state']==traces[0]['final_state']
        assert [x['new_positions'] for x in a['acceptances']]==[x['new_positions'] for x in traces[0]['acceptances']]
    for trace in traces[1:]:
        assert trace['final_state']==traces[0]['final_state']
        assert trace['acceptances']==traces[0]['acceptances']
    final=np.asarray(traces[0]['final_state']);plen=traces[0]['prompt_length']
    returned={int(p) for p in range(plen,len(final)) if final[p] not in (EOS_ID,MASK_ID)}
    assert [int(final[p]) for p in sorted(returned)]==results['trace'][0]['output_ids']
    first_accept={}
    for a in traces[0]['acceptances']:
        for p in a['new_positions']:
            assert p not in first_accept
            first_accept[p]=a['index']
    assert returned<=first_accept.keys()
    materialized={p for p in first_accept if p>=plen}
    eos=[p for p in materialized if final[p]==EOS_ID]
    first_eos=min(eos) if eos else len(final)
    contiguous={p for p in returned if p<first_eos}
    calls={e['call_id']:e for e in traces[0]['events']}
    groups=collections.defaultdict(list)
    for rank,t in enumerate(traces):
        for r in t['routes']:groups[(r['call_id'],r['layer'])].append((rank,r))
    acc={a['index']:a for a in traces[0]['acceptances']}
    iteration=collections.defaultdict(lambda:collections.defaultdict(float))
    counters=collections.Counter();denominator=collections.Counter();route_rows=[]
    edge_data={};load_data={};ownership_checks=0
    for (call,layer),records in sorted(groups.items()):
        assert len(records)==4
        e=calls[call];idx=e['iteration']['index'] if e['phase']=='denoise' else -1
        edges=np.zeros((4,4),dtype=np.int64);loads=np.zeros(4,dtype=np.int64);expect_recv=np.zeros(4,dtype=np.int64)
        received={};received_paths={};positions=[];experts=set();weighted=np.zeros(256,dtype=float)
        for rank,r in records:
            z=arrays[rank];tag=r['tag'];ids=z[tag+'_ids'];pos=z[tag+'_positions'].astype(int)
            before=z[tag+'_before'];after=z[tag+'_after'];weights=z[tag+'_weights'].astype(float)
            assert ids.shape==(len(pos),8) and np.all(np.isfinite(weights))
            # SGLang keeps top-k weights normalized here and applies the
            # checkpoint's routed_scaling_factor=2.5 inside the expert runner.
            assert np.allclose(weights.sum(axis=1),1,rtol=.02,atol=.01),'native top-k normalization'
            t=traffic(ids,rank,r['hidden_row_bytes']);n=len(pos);positions.extend(pos.tolist());experts.update(ids.flatten().tolist())
            np.add.at(weighted,ids.flatten(),weights.flatten())
            edges[rank]+=t['destinations'].sum(axis=0);expect_recv+=t['destinations'].sum(axis=0)
            loads+=np.bincount(t['owners'].flatten(),minlength=4)
            received[rank]=r['recv_rows'];received_paths[rank]=int((z[tag+'_recv_ids']>=0).sum())
            assert np.all((z[tag+'_recv_ids']>=-1)&(z[tag+'_recv_ids']<64)),'owner-local received IDs'
            output=(pos>=plen);live=before&output;new=live&~after;dead=output&~before;context=~output
            final_mask=np.array([p in returned for p in pos]);contiguous_mask=np.array([p in contiguous for p in pos])
            one=np.array([p in returned and idx==first_accept.get(p,-999) for p in pos])
            all_one=np.array([p in materialized and idx==first_accept.get(p,-999) for p in pos])
            masks={'actual':np.ones(n,dtype=bool),'actual_output':output,'context':context,'live_only':live,
                'dead_output':dead,'live_final_only':live&final_mask,'live_contiguous':live&contiguous_mask,
                'one_shot':one,'one_shot_all_materialized':all_one,'one_shot_contiguous':one&contiguous_mask,
                'unaccepted_after':live&after,'live_nonreturned':live&~final_mask}
            for name,mask in masks.items():
                value=int(t['total'][mask].sum());counters[name]+=value
                iteration[idx][name+'_bytes']+=value
            for component in ('dispatch','combine','remote_paths','local_paths'):
                iteration[idx][component]+=int(t[component].sum())
            iteration[idx]['routed_paths']+=int(ids.size)
            iteration[idx]['source_rows']+=n
            iteration[idx]['remote_fanout_sum']+=int(t['remote_destinations'].sum())
            iteration[idx]['live_rows_layer_sum']+=int(live.sum())
            iteration[idx]['new_rows_layer_sum']+=int(new.sum())
            for p in pos[one]:denominator[int(p)]+=1
            route_rows.append(dict(request=directory.name,call_id=call,iteration=idx,layer=layer,source_rank=rank,
                rows=n,remote_paths=int(t['remote_paths'].sum()),dispatch_bytes=int(t['dispatch'].sum()),
                combine_bytes=int(t['combine'].sum()),recv_rows=r['recv_rows'],unique_experts=int(len(np.unique(ids))),
                rank_assignments=np.bincount(t['owners'].flatten(),minlength=4).tolist(),
                remote_destinations=int(t['remote_destinations'].sum())))
        assert len(positions)==len(set(positions)),'source row partition is exact, not replicated DP users'
        assert all(received[r]==int(expect_recv[r]) for r in range(4)),'actual DeepEP receive buffers must match destination-dedup accounting'
        assert all(received_paths[r]==int(loads[r]) for r in range(4)),'received expert paths match physical ownership'
        ownership_checks+=1
        iteration[idx]['layers']+=1
        iteration[idx]['unique_experts_layer_sum']+=len(experts)
        iteration[idx]['active_experts_per_rank_layer_sum']=iteration[idx].get('active_experts_per_rank_layer_sum',np.zeros(4,dtype=int))+np.bincount(np.asarray(sorted(experts))//64,minlength=4)
        if idx not in edge_data:edge_data[idx]=np.zeros((4,4),dtype=np.int64);load_data[idx]=np.zeros(4,dtype=np.int64)
        edge_data[idx]+=edges;load_data[idx]+=loads
    assert set(denominator)==returned and all(n==19 for n in denominator.values()),'each final output acceptance route counted once in each19MoElayers'
    assert counters['actual']==counters['context']+counters['live_only']+counters['dead_output']
    assert counters['one_shot']>0
    rows=[];cum=0;hot=[]
    for idx,values in sorted(iteration.items()):
        if idx>=0:
            a=acc[idx];forward_positions=set()
            for call,layer in groups:
                if calls[call]['phase']=='denoise' and calls[call]['iteration']['index']==idx and layer==1:
                    for rank,r in groups[(call,layer)]:forward_positions.update(arrays[rank][r['tag']+'_positions'].tolist())
            live=set(a['live_positions'])&forward_positions
            newly=set(a['new_positions'])&forward_positions
            cum+=len(newly);yield_=len(newly)/len(live) if live else None
            values.update(live_tokens=len(live),newly_accepted_tokens=len(newly),already_accepted_routed_tokens=len([p for p in forward_positions if p>=plen and p not in live]),
                still_live_tokens=len(live-newly),acceptance_yield=yield_,cumulative_accepted=cum,
                status='ZERO_ACCEPT_ITERATION' if not newly else 'ACCEPT',native_iter=a['native_iter'],block=a['block'],
                start=a['start'],end=a['end'],is_cross_block=a['is_cross_block'])
            hot.append(int(load_data[idx].argmax()))
        else:values.update(status='PROMPT_PREFILL',live_tokens=0,newly_accepted_tokens=0,acceptance_yield=None,cumulative_accepted=0)
        values.update(graph_stats(edge_data[idx],load_data[idx]))
        values['rank_assignments']=load_data[idx].tolist();values['rank_edges']=edge_data[idx].tolist()
        values['bytes_per_accepted_token']=values['actual_bytes']/values['newly_accepted_tokens'] if values['newly_accepted_tokens'] else None
        values['live_unaccepted_traffic_fraction']=values['unaccepted_after_bytes']/values['live_only_bytes'] if values['live_only_bytes'] else None
        values['remote_fanout_per_row']=values['remote_fanout_sum']/values['source_rows']
        values['routed_rows_per_layer']=values['source_rows']/values['layers']
        values['unique_experts_per_layer']=values['unique_experts_layer_sum']/values['layers']
        values['active_experts_per_rank_mean']=(values['active_experts_per_rank_layer_sum']/values['layers']).tolist()
        del values['active_experts_per_rank_layer_sum']
        rows.append(dict(request=directory.name,iteration=idx,**values))
    timing=[read(directory/f'timing.rank{r}.json') for r in range(4)]
    sums=[]
    for t in timing:
        denoise_calls={e['call_id'] for e in t['events'] if e['phase']=='denoise'}
        out=collections.defaultdict(float)
        for s in t['stages']:
            if s['call_id'] in denoise_calls:out[s['stage']+'_ms']+=s['ms']
        out['denoising_forward_ms']=sum(e['forward_ms'] for e in t['events'] if e['call_id'] in denoise_calls)
        out['denoising_span_ms']=t.get('denoise_span_ms')
        sums.append(dict(out))
    critical=max(range(4),key=lambda r:sums[r]['denoising_forward_ms'])
    stage=sums[critical];comm=stage['dispatch_ms']+stage['combine_ms']
    overhead=results['timing'][0]['wall_ms']/results['bypass'][0]['wall_ms']-1
    trace_overhead=results['trace'][0]['wall_ms']/results['bypass'][0]['wall_ms']-1
    result=dict(request=directory.name,request_id=results['trace'][0]['request_id'],domain=results['trace'][0]['domain'],
        block_length=results['trace'][0]['block_length'],prompt_tokens=plen,output_tokens=len(returned),
        contiguous_output_tokens=len(contiguous),materialized_accepted_tokens=len(materialized),output_contains_mask=False,
        finish='EOS_STOP' if eos else 'BOUNDED_BLOCK_CAP',nfe=results['trace'][0]['nfe'],iterations=len(acc),
        zero_accept_iterations=sum(a['status']=='ZERO_ACCEPT_ITERATION' for a in rows),
        **{f'B_{k}':int(v) for k,v in counters.items()},Raw_Amp=counters['actual']/counters['one_shot'],
        Raw_output_Amp=counters['actual_output']/counters['one_shot'],Residual_Amp=counters['live_only']/counters['one_shot'],
        Residual_final_positions_Amp=counters['live_final_only']/counters['one_shot'],
        Residual_all_materialized_Amp=counters['live_only']/counters['one_shot_all_materialized'],
        Residual_contiguous_Amp=counters['live_contiguous']/counters['one_shot_contiguous'] if counters['one_shot_contiguous'] else None,
        actual_bytes_per_output=counters['actual']/len(returned),output_bytes_per_output=counters['actual_output']/len(returned),
        dead_share_actual=counters['dead_output']/counters['actual'],context_share_actual=counters['context']/counters['actual'],
        dead_share_output=counters['dead_output']/counters['actual_output'],
        dead_explained_output_excess=counters['dead_output']/(counters['actual_output']-counters['one_shot']),
        repeated_live_excess=counters['live_only']-counters['one_shot'],
        live_unaccepted_fraction=counters['unaccepted_after']/counters['live_only'],
        critical_rank=critical,**stage,comm_ms=comm,comm_share_denoising_forward=comm/stage['denoising_forward_ms'],
        comm_share_denoising_span=comm/stage['denoising_span_ms'] if stage['denoising_span_ms'] else None,
        comm_share_request_wall=comm/results['timing'][0]['wall_ms'],
        bypass_wall_ms=results['bypass'][0]['wall_ms'],timing_wall_ms=results['timing'][0]['wall_ms'],
        timing_overhead_fraction=overhead,trace_overhead_fraction=trace_overhead,semantic_parity=parity,
        physical_buffer_validation_checks=ownership_checks,rank_edge_density=np.mean([r['edge_density'] for r in rows if r['iteration']>=0]),
        hottest_rank_switch_rate=np.mean(np.array(hot)[1:]!=np.array(hot)[:-1]) if len(hot)>1 else 0,
        rank_imbalance_iteration_variance=np.var([r['rank_max_mean'] for r in rows if r['iteration']>=0]))
    for z in arrays:z.close()
    return result,rows,route_rows,dict(critical_rank_rule='one rank with greatest cumulative denoising forward CUDA time; all component sums on that same rank',rank_sums=sums)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=ROOT/'raw'/a.run;out=ROOT/'results'/a.output
    if out.exists():raise FileExistsError(out)
    requests=[];iterations=[];route_rows=[];timing={}
    for directory in sorted(source.glob('*_b*')):
        r,i,t,s=request(directory);requests.append(r);iterations.extend(i);route_rows.extend(t);timing[r['request']]=s
    assert requests
    out.mkdir(parents=True)
    pd.DataFrame(requests).to_csv(out/'requests.csv',index=False)
    pd.DataFrame(iterations).to_csv(out/'iterations.csv',index=False)
    pd.DataFrame(route_rows).to_csv(out/'route_accounting.csv',index=False)
    write(out/'timing_ledger.json',timing)
    keys=['Raw_Amp','Raw_output_Amp','Residual_Amp','Residual_final_positions_Amp','Residual_all_materialized_Amp',
        'Residual_contiguous_Amp','actual_bytes_per_output','output_bytes_per_output','dead_share_output',
        'dead_explained_output_excess','live_unaccepted_fraction','comm_share_denoising_forward','comm_share_denoising_span','comm_share_request_wall',
        'timing_overhead_fraction','trace_overhead_fraction','rank_edge_density','hottest_rank_switch_rate']
    df=pd.DataFrame(requests);summary={}
    for block in sorted(df.block_length.unique()):
        blockrows=df[df.block_length==block]
        for domain in ('all','gsm8k','humaneval'):
            subset=blockrows if domain=='all' else blockrows[blockrows.domain==domain]
            if len(subset):summary[f'b{block}_{domain}']={k:bootstrap_mean(subset[k]) for k in keys}
    primary=df[df.block_length==32];stat=summary['b32_all']
    timing_valid=abs(stat['timing_overhead_fraction']['mean'])<=.10 and bool(primary.semantic_parity.all())
    # Robustness: the conservative final-position-only accounting must also pass.
    residual=stat['Residual_final_positions_Amp']['mean']
    # Request wall includes decoder, cache updates AND initial prompt prefill.
    # This is a conservative lower bound on communication/denoising share;
    # forward-only CUDA intervals omit acceptance/controller overhead.
    share=stat['comm_share_request_wall']['mean']
    verdict=decision(residual,share,timing_valid)
    if verdict=='GO' and (stat['Residual_final_positions_Amp']['ci95'][0]<2 or
            stat['comm_share_request_wall']['ci95'][0]<.15):verdict='CONDITIONAL'
    payload=dict(verdict=verdict,stats=summary,timing_valid=timing_valid,
        request_count=len(primary),secondary_count=len(df)-len(primary),
        physical_ep4_validated=True,semantic_parity=True,
        accounting_label='backend-validated destination-deduplicated logical remote hidden/output payload bytes, not NIC wire counters',
        live_only_label='optimistic structural accounting, not measured executable speedup',
        one_shot_label='acceptance-route structural reference, not an AR comparison',
        conservative_gate='final-returned-position-only residual removes eventual EOS/tail inflation',
        timing_gate_denominator='entire timing-pass request wall (includes prompt prefill); conservative lower bound on communication/total denoising latency',
        gates=dict(residual_pass=residual>=2,communication_pass=share>=.15,timing_valid=timing_valid,
            residual_ci_pass=stat['Residual_final_positions_Amp']['ci95'][0]>=2,
            conservative_communication_ci_pass=stat['comm_share_request_wall']['ci95'][0]>=.15),
        bootstrap='2000 request-level resamples independently within each block/domain; no iteration/layer replication')
    write(out/'summary.json',payload)
    print(json.dumps(dict(verdict=verdict,timing_valid=timing_valid,primary=stat),indent=2))

if __name__=='__main__':main()
