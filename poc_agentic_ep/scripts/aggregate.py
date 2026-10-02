"""Audit raw captured positions and export compact new-only route evidence."""
import argparse
import json
import numpy as np
import pandas as pd
from poc_agentic_ep.contract import ROOT,write,sha
from poc_agentic_ep.metrics import source_counts,traffic


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--version',required=True);a=p.parse_args()
    run=ROOT/'raw'/a.run;out=ROOT/'results'/a.version
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);raw=out/'raw_routing';raw.mkdir()
    frozen=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text())
    lookup={x['request_id']:x for x in frozen['rows']}
    workers=json.loads((run/'worker.dp0.json').read_text())['records']
    frames=[{(f['wave'],f['forward']):f for f in map(json.loads,(run/f'frames.rank{r}.jsonl').read_text().splitlines())} for r in range(4)]
    all_counts=[];weights_mass=[];layer_rows=[];turn_rows=[];audits=[];request_ids=[];timings=[];weight_bytes=set()
    for record in workers:
        rid=record['request_id']
        if rid not in lookup:continue
        row=lookup[rid];wave=record['wave']
        semantic=[f for (w,_),f in frames[0].items() if w==wave and f.get('sampled_tokens') is not None]
        assert len(semantic)==1, 'single teacher-forced prefill invocation required'
        forward=semantic[0]['forward']
        selected=[frames[r][(wave,forward)] for r in range(4)]
        for f in selected:
            for layer in f['layers']:weight_bytes.add(layer['expert_weight_bytes_each'])
        handles=[np.load(run/f'routes.wave{wave}.rank{r}.npz') for r in range(4)]
        ids_by_rank=[];weights_by_rank=[];positions=[];sources=[]
        for rank in range(4):
            layers=selected[rank]['layers'];assert len(layers)==48
            idarr=np.stack([handles[rank][l['route_key']+'_ids'] for l in layers])
            warr=np.stack([handles[rank][l['route_key']+'_weights'] for l in layers])
            n=layers[0]['captured_rows'];start=layers[0]['captured_position_start']
            assert all(l['captured_rows']==n and l['captured_position_start']==start for l in layers)
            assert idarr.shape==(48,n,8) and (idarr>=0).all() and (idarr<128).all()
            assert np.allclose(warr.sum(axis=-1),1,atol=2e-3)
            if rank>=2:assert n==0, 'idle DP must not become a replicated request'
            ids_by_rank.append(idarr.astype(np.uint8));weights_by_rank.append(warr.astype(np.float32))
            positions.extend(range(start,start+n));sources.extend([rank]*n)
        ids=np.concatenate(ids_by_rank,axis=1);weights=np.concatenate(weights_by_rank,axis=1)
        positions=np.array(positions,dtype=np.uint16);sources=np.array(sources,dtype=np.uint8)
        order=np.argsort(positions);ids=ids[:,order];weights=weights[:,order];positions=positions[order];sources=sources[order]
        assert len(set(map(int,positions)))==len(positions),'duplicate TP source copy'
        assert (positions>=row['exclusion_start']).all(),'shared prefix leaked'
        decode=positions>=row['prompt_tokens'];assert decode.sum()==row['decode_tokens']
        if len(positions):assert np.array_equal(positions,np.arange(int(positions[0]),row['prompt_tokens']+row['decode_tokens']))
        np.savez_compressed(raw/(rid+'.npz'),expert_ids=ids,routing_weights=weights,positions=positions,
            source_ep_rank=sources,token_category=decode.astype(np.uint8))
        phase_counts=[];phase_weights=[]
        for phase,mask in [('new_prefill',~decode),('decode',decode)]:
            sc=source_counts(ids[:,mask,:],sources[mask]);counts=sc.sum(axis=1)
            weighted=np.zeros((48,128))
            for layer in range(48):weighted[layer]=np.bincount(ids[layer,mask,:].ravel(),weights=weights[layer,mask,:].ravel(),minlength=128)
            phase_counts.append(sc);phase_weights.append(weighted)
            load=counts.reshape(48,4,32).sum(axis=-1)
            counters=traffic(ids[:,mask,:],sources[mask])
            turn_rows.append(dict(request_id=rid,trajectory_id=row['trajectory_id'],turn=row['turn'],split=row['split'],phase=phase,
                tokens=int(mask.sum()),prompt_tokens=row['prompt_tokens'],common_prompt_prefix=row['common_prompt_prefix'],
                previous_completed_prefix=row['previous_completed_prefix'],exclusion_start=row['exclusion_start'],
                expert_counts=json.dumps(counts.sum(axis=0).tolist()),rank_counts=json.dumps(load.sum(axis=0).tolist()),
                remote_paths=counters['remote_paths'],remote_dispatch_bytes=counters['remote_dispatch_bytes'],remote_combine_bytes=counters['remote_combine_bytes'],
                rank_max_mean=counters['rank_max_mean'],rank_cv=counters['rank_cv'],hottest_rank=int(load.sum(axis=0).argmax())))
            for layer in range(48):
                layer_rows.append(dict(request_id=rid,trajectory_id=row['trajectory_id'],turn=row['turn'],split=row['split'],phase=phase,layer=layer,
                    tokens=int(mask.sum()),paths=int(counts[layer].sum()),rank_counts=json.dumps(load[layer].tolist()),
                    active_experts=int((counts[layer]>0).sum()),max_rank_load=int(load[layer].max()),
                    rank_max_mean=float(load[layer].max()/max(load[layer].mean(),1)),rank_cv=float(load[layer].std()/max(load[layer].mean(),1)),
                    expert_counts=json.dumps(counts[layer].tolist()),routing_weight_mass=json.dumps(weighted[layer].tolist())))
        all_counts.append(phase_counts);weights_mass.append(phase_weights);request_ids.append(rid)
        query_start=selected[0]['layers'][0]['actual_query_start']
        audits.append(dict(request_id=rid,captured=int(len(positions)),new_prefill=int((~decode).sum()),decode=int(decode.sum()),
            actual_cached_prefix=query_start,common_prompt_prefix=row['common_prompt_prefix'],
            stronger_completed_prefix=row['exclusion_start'],minimum_captured_position=int(positions.min()),
            expected_start=max(query_start,row['exclusion_start']),common_prefix_in_primary=False))
        # One cumulative-critical rank across48 layers; trace timings diagnostic.
        critical=max(range(4),key=lambda r:sum(x['moe_ms'] for x in selected[r]['layers']))
        for layer,value in enumerate(selected[critical]['layers']):
            timings.append(dict(request_id=rid,critical_rank=critical,layer=layer,
                **{k:value[k] for k in ('dispatch_ms','expert_ms','combine_ms','moe_ms')}))
        for h in handles:h.close()
    np.savez_compressed(out/'aggregated.npz',source_expert_counts=np.asarray(all_counts),weight_mass=np.asarray(weights_mass),request_ids=np.asarray(request_ids))
    pd.DataFrame(turn_rows).to_csv(out/'turn_metrics.csv',index=False)
    pd.DataFrame(layer_rows).to_csv(out/'layer_metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'diagnostic_moe_timing.csv',index=False)
    write(out/'prefix_exclusion_audit.json',dict(status='PASS',rows=audits,rule='only positions >= max(recorded completed frontier, actual KV-cache query start)',
        raw_route_hashes={p.name:sha(p) for p in raw.glob('*.npz')},manifest_sha256=sha(ROOT/'results/frozen_manifest_v2.json')))
    for rank in range(4):
        proof=json.loads((run/f'ownership.rank{rank}.json').read_text())
        expected=[-1]*128;expected[rank*32:(rank+1)*32]=list(range(32));assert proof['expert_map']==expected
        write(out/f'ownership.rank{rank}.json',proof)
        write(out/f'runtime_memory.rank{rank}.json',json.loads((run/f'runtime_memory.rank{rank}.json').read_text()))
    write(out/'p2p.json',json.loads((run/'p2p.json').read_text()))
    assert len(weight_bytes)==1
    write(out/'loaded_expert_signature.json',dict(bytes_per_expert_per_layer=next(iter(weight_bytes)),
        layers=48,experts=128,top_k=8,hidden=2048,intermediate=768,dtype='bfloat16',
        source='fresh loaded w1.numel()*element_size + w2, divided by32 local experts; all48 layers and4 ranks checked',
        shapes=[[32,1536,2048],[32,2048,768]],not_actual_hbm_traffic=True))
    print(json.dumps(dict(status='PASS',turns=len(request_ids),raw_only_new_and_decode=True)))


if __name__=='__main__':main()
