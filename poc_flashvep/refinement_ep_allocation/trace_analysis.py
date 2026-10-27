"""Analyze real dispatch buffers. Do not call logical bytes wire traffic."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import ROOT, write

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',default='A_sanity')
    a=p.parse_args()
    base=ROOT/'ep4'/a.run_id
    requests=[]
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or not (folder/'trace.rank0.json').exists():
            continue
        matrices={}
        measured=[]
        request={}
        for rank in range(4):
            meta=json.loads((folder/f'trace.rank{rank}.json').read_text())
            data=np.load(folder/f'trace.rank{rank}.npz')
            remote_rows=0
            counts=np.zeros(4,dtype=np.int64)
            routed_edges=0
            recv_rows=0
            for route in meta['routes']:
                tag=route['tag']
                ids=data[tag+'_ids'].astype(np.int64)
                owners=ids//64
                routed_edges+=ids.size
                counts+=np.bincount(owners.flatten(),minlength=4)
                recv_rows+=route['recv_rows']
                for dest in range(4):
                    if dest!=rank:
                        remote_rows+=int(np.any(owners==dest,axis=1).sum())
                key=(route['call_id'],route['layer'])
                matrices.setdefault(key,[]).append((rank,ids, data[tag+'_weights'],data[tag+'_before']))
                local_ids=data[tag+'_recv_ids'].astype(np.int64)
                real=local_ids[local_ids>=0]
                assert np.all(real<64)
                route['actual_received_expert_edges']=int(real.size)
            stage={s:sum(x['ms'] for x in meta['stages'] if x['stage']==s)
                   for s in ('router','dispatch','expert','combine')}
            result=json.loads((folder/f'trace.result.rank{rank}.json').read_text())
            fwd=sum(x['forward_ms'] for x in meta['events'])
            measured.append(dict(rank=rank, actual_routed_edges=routed_edges,
                owner_edge_counts=counts.tolist(), actual_receive_rows=recv_rows,
                observed_dispatch_calls=len(meta['routes']), remote_token_rank_pairs=remote_rows,
                logical_dispatch_payload_bytes=remote_rows*4096,
                logical_combine_payload_bytes=remote_rows*4096,
                traffic_kind='ESTIMATED_BYTES_FROM_MEASURED_ROUTING_NOT_PHYSICAL_WIRE_BYTES',
                observed_stage_ms=stage, observed_forward_ms=fwd, observer_wall_ms=result['wall_ms'],
                stage_time_kind='MEASURED_GPU_OBSERVER_RANK_LOCAL_NOT_GLOBAL_CRITICAL_PATH',
                observed_stage_wall_percent={s:100*t/result['wall_ms'] for s,t in stage.items()}))
        load=np.sum([m['owner_edge_counts'] for m in measured],axis=0)
        baseline=json.loads((folder/'bypass.result.rank0.json').read_text())
        noop=json.loads((folder/'hooked_noop.result.rank0.json').read_text())
        request.update(sample_id=baseline['sample_id'], nfe=baseline['nfe'],
            no_op_token_parity=baseline['output_ids']==noop['output_ids'],
            no_op_nfe_parity=baseline['nfe']==noop['nfe'],
            observer_token_parity=baseline['output_ids']==json.loads((folder/'trace.result.rank0.json').read_text())['output_ids'],
            rank_metrics=measured, total_owner_edge_counts=load.tolist(),
            owner_work_cv=float(load.std()/load.mean()), owner_max_mean=float(load.max()/load.mean()),
            request_latency_primary_status='NOT_MEASURED_ISOLATED',
            bypass_wall_ms=baseline['wall_ms'], no_op_wall_ms=noop['wall_ms'])
        # A structural bound only. An edge removal does not necessarily remove a token-rank message.
        deletion=[]
        for fraction in (.05,.10,.15,.20):
            edge_removed=0
            remote_before=0
            remote_after=0
            unique_before=unique_after=0
            padded_before=padded_after=0
            for key,parts in matrices.items():
                ids=np.concatenate([v[1] for v in sorted(parts,key=lambda v:v[0])])
                weights=np.concatenate([v[2] for v in sorted(parts,key=lambda v:v[0])])
                source=np.concatenate([np.full(len(v[1]),v[0]) for v in sorted(parts,key=lambda v:v[0])])
                flat_order=np.argsort(weights.flatten(),kind='stable')
                removed=np.zeros(ids.shape,dtype=bool)
                per_token=np.zeros(len(ids),dtype=np.int64)
                target=int(ids.size*fraction)
                for index in flat_order:
                    row,slot=divmod(int(index),8)
                    if per_token[row]<7:
                        removed[row,slot]=True
                        per_token[row]+=1
                        edge_removed+=1
                        target-=1
                        if target==0:
                            break
                owners=ids//64
                for dest in range(4):
                    eligible=source!=dest
                    remote_before+=int((np.any(owners==dest,axis=1)&eligible).sum())
                    remote_after+=int((np.any((owners==dest)&~removed,axis=1)&eligible).sum())
                    hist_before=np.bincount(ids[owners==dest]-dest*64,minlength=64)
                    hist_after=np.bincount(ids[(owners==dest)&~removed]-dest*64,minlength=64)
                    unique_before+=int((hist_before>0).sum())
                    unique_after+=int((hist_after>0).sum())
                    rows_before=int(np.any(owners==dest,axis=1).sum())
                    rows_after=int(np.any((owners==dest)&~removed,axis=1).sum())
                    # Source-confirmed fallback configuration. This is structural accounting,
                    # not a measured number of executed instructions or memory transactions.
                    tile_before=16 if rows_before<=64 else 64
                    tile_after=16 if rows_after<=64 else 64
                    padded_before+=int(((hist_before+tile_before-1)//tile_before*tile_before).sum())
                    padded_after+=int(((hist_after+tile_after-1)//tile_after*tile_after).sum())
            deletion.append(dict(policy='ROUTER_WEIGHT_ONLY_STRUCTURAL_COUNTERFACTUAL',
                target_edge_reduction=fraction, removed_edges=edge_removed,
                actual_edge_reduction=edge_removed/int(load.sum()),
                remote_pair_reduction=1-remote_after/remote_before if remote_before else 0,
                unique_active_expert_invocation_reduction=1-unique_after/unique_before if unique_before else 0,
                padded_expert_rows_reduction=1-padded_after/padded_before if padded_before else 0,
                padded_work_kind='SOURCE_DERIVED_TILE_ACCOUNTING_NOT_MEASURED_KERNEL_WORK',
                interpretation='TRACE_ESTIMATE_ONLY. No altered quality or physical sparse execution measured.'))
        request['deduplicated_traffic_counterfactuals']=deletion
        requests.append(request)
    target=ROOT/('ep_trace_audit.json' if a.run_id=='A_sanity' else f'ep_trace_audit.{a.run_id}.json')
    write(target,dict(run_id=a.run_id, requests=requests,
        evidence_scope='Fresh two-request sanity traces. Not a quality or speedup claim.'))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('rank_metrics',)} for r in requests],indent=2))

if __name__=='__main__':
    main()
