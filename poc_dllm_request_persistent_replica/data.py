"""Physical source/live-scope representation for efficient immutable-plan evaluation."""
import collections
import numpy as np
from poc_dllm_block_local_replica.metrics import closure_groups,post_traffic
from poc_dllm_request_persistent_replica.contract import *


class Profile:
    def __init__(self,blocks):
        self.blocks=blocks
        self._frequency=None;self._groups=None
        self.nblocks=len(blocks)
        self.iterations=[b['iterations'] for b in blocks]
        self.n=sum(self.iterations)
        self.records={};self.counts=np.zeros((self.n,19,4,256),dtype=np.int16)
        offset=0
        for ordinal,b in enumerate(blocks):
            for (t,l,s),ids in b['data'].items():
                key=(l,s)
                entry=self.records.setdefault(key,dict(ids=[],time=[],block=[]))
                entry['ids'].append(ids)
                entry['time'].append(np.full(len(ids),offset+t,dtype=np.int32))
                entry['block'].append(np.full(len(ids),ordinal,dtype=np.int16))
                self.counts[offset+t,l-1,s]=np.bincount(ids.flatten(),minlength=256)
            offset+=b['iterations']
        for entry in self.records.values():
            for k in ('ids','time','block'):
                entry[k]=np.concatenate(entry[k]) if entry[k] else np.empty((0,8) if k=='ids' else 0,dtype=np.int16)
        self.load=np.zeros((self.n,19,4),dtype=np.int32)
        for owner in range(4):self.load[:,:,owner]=self.counts[:,:,:,owner*64:(owner+1)*64].sum(axis=(2,3))

    def subset(self,first=0,last=None):
        return Profile(self.blocks[first:last])

    def frequency(self):
        if self._frequency is None:self._frequency=self.counts.sum(axis=0,dtype=np.int64)
        return self._frequency

    def groups(self):
        if self._groups is not None:return self._groups
        result={s:{l:collections.Counter() for l in range(1,20)} for s in range(4)}
        for (l,s),entry in self.records.items():result[s][l].update(closure_groups(entry['ids'],s))
        self._groups=result
        return result

    def evaluate(self,plan,profile_blocks=0):
        masks=np.zeros((19,4,256),dtype=bool)
        edges=np.zeros((4,4),dtype=np.int64)
        for s,layers in plan.items():
            for l,experts in layers.items():
                for e in experts:
                    assert e//64!=s
                    masks[l-1,s,e]=True;edges[e//64,s]+=EXPERT_BYTES
        postload=np.zeros_like(self.load,dtype=np.int64)
        dc=np.zeros(self.nblocks,dtype=np.int64);base=np.zeros_like(dc)
        pertime_saved=np.zeros(self.n,dtype=np.int64);replica_use=[]
        for (l,s),r in self.records.items():
            ids=r['ids'];original=ids//64;owner=original.copy()
            chosen=masks[l-1,s,ids] & (r['block'][:,None]>=profile_blocks)
            owner[chosen]=s
            original_dc=np.stack([(original==d).any(axis=1) for d in range(4)],axis=1)
            modified_dc=np.stack([(owner==d).any(axis=1) for d in range(4)],axis=1)
            original_dc[:,s]=False;modified_dc[:,s]=False
            a=original_dc.sum(axis=1)*ROW_DC_BYTES;p=modified_dc.sum(axis=1)*ROW_DC_BYTES
            base+=np.bincount(r['block'],weights=a,minlength=self.nblocks).astype(np.int64)
            dc+=np.bincount(r['block'],weights=p,minlength=self.nblocks).astype(np.int64)
            pertime_saved+=np.bincount(r['time'],weights=a-p,minlength=self.n).astype(np.int64)
            for d in range(4):postload[:,l-1,d]+=np.bincount(r['time'],weights=(owner==d).sum(axis=1),minlength=self.n).astype(np.int64)
            for e in plan.get(s,{}).get(l,set()):
                use=np.any((ids==e)&(r['block'][:,None]>=profile_blocks),axis=1)
                replica_use.append(dict(layer=l,source=s,expert=e,owner=e//64,
                    future_blocks_used=len(set(r['block'][use].tolist())),future_iterations_used=len(set(r['time'][use].tolist()))))
        assert postload.sum()==self.load.sum(),'placement does not remove expert paths'
        original_max=self.load.max(axis=2);newmax=postload.max(axis=2);nz=original_max>0
        ratios=np.divide(newmax,original_max,out=np.ones_like(newmax,dtype=float),where=nz)-1
        wt,wl=np.unravel_index(np.argmax(ratios),ratios.shape)
        totalbefore=self.load.sum(axis=(0,1));totalafter=postload.sum(axis=(0,1))
        mean=totalafter.mean();peakextra=edges.sum(axis=0).max()
        cumulative=np.cumsum(pertime_saved);break_even=np.flatnonzero(cumulative>=edges.sum()) if edges.sum() else np.array([],dtype=int)
        offsets=np.cumsum([0]+self.iterations)
        return dict(live_bytes=int(base.sum()),post_bytes=int(dc.sum()),saved_bytes=int((base-dc).sum()),
            block_baseline_bytes=base.tolist(),block_post_bytes=dc.tolist(),per_iteration_saved_bytes=pertime_saved.tolist(),
            copy_bytes=int(edges.sum()),initial_copy_bytes=int(edges.sum()),update_bytes=0,
            copy_edges=edges.tolist(),peak_memory_gib=peakextra/1024**3,
            future_blocks=max(self.nblocks-profile_blocks,0),future_iterations=sum(self.iterations[profile_blocks:]),
            profile_blocks=min(profile_blocks,self.nblocks),
            baseline_rank_assignments=totalbefore.tolist(),rank_assignments=totalafter.tolist(),
            max_rank_load_increase=float(totalafter.max()/totalbefore.max()-1) if totalbefore.max() else 0.,
            worst_stage_load_increase=float((newmax[nz]/original_max[nz]-1).max()) if nz.any() else 0.,
            worst_stage_detail=dict(iteration=int(wt),layer=int(wl+1),
                original_assignments=self.load[wt,wl].tolist(),replica_assignments=postload[wt,wl].tolist(),
                original_max=int(original_max[wt,wl]),replica_max=int(newmax[wt,wl])),
            stage_critical_load_increase=float(newmax.sum()/original_max.sum()-1) if original_max.sum() else 0.,
            rank_max_mean=float(totalafter.max()/mean) if mean else 0.,rank_cv=float(totalafter.std()/mean) if mean else 0.,
            hottest=int(totalafter.argmax()),byte_roi=float((base-dc).sum()/edges.sum()) if edges.sum() else None,
            byte_break_even_iteration=int(break_even[0]) if len(break_even) else None,
            byte_break_even_block=int(np.searchsorted(offsets[1:],break_even[0],side='right')) if len(break_even) else None,
            replica_use=replica_use)


def load_profile(info):
    blocks=[]
    archive=Path(info['archive'])
    if not archive.exists():
        # Committed manifests retain original provenance; a reproduced run
        # rebuilds derived archives under its own artifact root.
        archive=ROOT/'data/live_routes'/archive.name
    with np.load(archive) as z:
        for b in info['blocks']:
            data={}
            for t in range(b['iterations']):
                for l in range(1,20):
                    for s in range(4):
                        key=f"b{b['block']}_t{t}_l{l}_s{s}_ids"
                        data[t,l,s]=z[key] if key in z else np.empty((0,8),dtype=np.int16)
            blocks.append(dict(iterations=b['iterations'],data=data,native_block=b['block']))
    return Profile(blocks)
