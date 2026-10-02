"""Lossless compact export: native logs/events, selected routed IDs/weights.

Model weights and hidden states are never exported. Original versioned raw
directories remain untouched. NPZ ZIP member overhead is removed by stacking
actual post-prefill verification routes; dummy/padding routes are retained.
"""
import argparse
import gzip
import json
import shutil

import numpy as np
import pandas as pd

from poc_speculative_moe_ep.contract import ROOT,SPEC,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True)
    p.add_argument('--trace-version',default='phase_a_v1');p.add_argument('runs',nargs='+');a=p.parse_args()
    root=ROOT/'artifacts'/a.version
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'authoritative_spec_snapshot.md').write_text(SPEC.read_text())
    table=pd.read_parquet(ROOT/'artifacts'/a.trace_version/'verification_metrics.parquet')
    originals=[]
    for name in a.runs:
        src=ROOT/'raw'/name;out=root/name;out.mkdir()
        for file in sorted(src.glob('*.json')):
            shutil.copyfile(file,out/file.name)
            originals.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file),export=str((out/file.name).relative_to(root))))
        for file in sorted(src.glob('frames.rank*.jsonl')):
            target=out/(file.name+'.gz')
            with file.open('rt') as inp,gzip.open(target,'wt',compresslevel=6) as dest:
                for line in inp:
                    row=json.loads(line)
                    # Raw numerical event endpoints/durations retained; remove
                    # repeated static tensor shapes and backend names already
                    # proven by runtime owner manifests. No rounding.
                    row['layers']=[{k:v for k,v in l.items() if k not in ('expert_shapes','prepare_backend','expert_backend','expert_weight_bytes_each')} for l in row['layers']]
                    dest.write(json.dumps(row,separators=(',',':'))+'\n')
            originals.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file),export=str(target.relative_to(root))))
        log=ROOT/'raw'/(name+'.log')
        if log.exists():
            with log.open('rb') as inp,gzip.open(out/'native.log.gz','wb',compresslevel=6) as dest:shutil.copyfileobj(inp,dest)
            originals.append(dict(path=str(log.relative_to(ROOT)),sha256=sha(log),export=str((out/'native.log.gz').relative_to(root))))
        snapshot=ROOT/'raw'/(name+'.gpu_session.json')
        if snapshot.exists():shutil.copyfile(snapshot,out/'gpu_session.json')
        selected=table[table.run==name]
        if selected.empty and list(src.glob('routes.wave*.rank0.npz')):
            # New route-repeat runs are not in the original trace table.
            measured={r['wave']:r for r in json.loads((src/'worker.dp0.json').read_text())['records'] if r['repeat']>=0}
            chosen=[]
            for wave,record in measured.items():
                frames=[f for f in map(json.loads,(src/'frames.rank0.jsonl').read_text().splitlines()) if f['wave']==wave and f.get('sampled_tokens') is not None]
                chosen.extend(dict(wave=wave,forward=f['forward'],request_id=record['request_id']) for f in frames[1:])
            selected=pd.DataFrame(chosen)
        if selected.empty:continue
        arrays={};indices=[]
        frame_maps=[{(f['wave'],f['forward']):f for f in map(json.loads,(src/f'frames.rank{r}.jsonl').read_text().splitlines())} for r in range(4)]
        for wave,g in selected.groupby('wave'):
            ordered=g.sort_values('forward');forwards=ordered.forward.tolist()
            meta=dict(wave=int(wave),forwards=forwards,request_id=ordered.request_id.iloc[0],ranks=[],
                accepted=[len([x for x in frame_maps[0][(int(wave),f)]['sampled_tokens'][0] if x>=0])-1 for f in forwards])
            for rank in range(4):
                original=src/f'routes.wave{wave}.rank{rank}.npz'
                originals.append(dict(path=str(original.relative_to(ROOT)),sha256=sha(original),
                    export='packed_routes.npz',scope='losslessly selected actual verification route arrays; prefill/warmup arrays remain local'))
                with np.load(original) as capture:
                    frames=[frame_maps[rank][(int(wave),f)] for f in forwards]
                    sizes=[f['layers'][0]['source_rows'] for f in frames];width=max(sizes)
                    ids=np.full((len(frames),48,width,8),-1,dtype=np.int16)
                    weights=np.zeros((len(frames),48,width,8),dtype=np.float32)
                    valid=[]
                    for i,frame in enumerate(frames):
                        valid.append(frame['layers'][0]['valid_source_rows'] if rank<2 else 0)
                        for layer,l in enumerate(frame['layers']):
                            assert l['source_rows']==sizes[i]
                            key=l['route_key'];actual=capture[key+'_ids']
                            ids[i,layer,:sizes[i]]=actual
                            weights[i,layer,:sizes[i]]=capture[key+'_weights']
                    key=f'w{wave}_r{rank}'
                    arrays[key+'_ids']=ids;arrays[key+'_weights']=weights
                    meta['ranks'].append(dict(rank=rank,key=key,source_rows=sizes,valid_source_rows=valid))
            indices.append(meta)
        np.savez_compressed(out/'packed_routes.npz',**arrays)
        write(out/'route_index.json',dict(trace_version=a.trace_version,waves=indices,
            scope='all selected verification IDs and FP32 weights, all48 layers/all4 actual ranks; no hidden vectors'))
    exports=[dict(path=str(f.relative_to(root)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(root.rglob('*')) if f.is_file()]
    write(root/'manifest.json',dict(exports=exports,originals=originals,spec_sha256=sha(SPEC),
        format='timing JSON loses only repeated static layer shape/backend labels; all numerical timings/counts preserved without rounding',
        raw_local_root=str(ROOT/'raw'),contains_model_weights=False,contains_hidden_states=False))
    print(json.dumps(dict(path=str(root),files=len(exports),bytes=sum(x['bytes'] for x in exports))))


if __name__=='__main__':main()
