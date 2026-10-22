"""Targeted first-divergence captures. Timings here are not serving results."""
import gc
import hashlib
import json
import torch
from .common import ROOT, write, digest

class Capture:
    def __init__(self,model,e,rid,stop_index=8,layers=(0,5,11,17,23),kv_steps=(0,8),forced_prefixes=None,capture_attention=False):
        self.model,self.e,self.rid=model,e,rid
        self.stop_index=stop_index
        self.records=[]
        self.tensors={}
        self.context=None
        self.token_index=0
        self.kv_steps=kv_steps
        self.handles=[]
        self.forced_prefixes=forced_prefixes
        self.coda_ids=[]
        self.sample_index=0
        self.sample_counts={}
        self.original_sample=e.model_runner.sampler.sample
        self.original_attend=e.cache_manager._attend_prepared
        self.original_core=model.recurrent_prepared
        self.original_forward=e.model_runner.runtime.forward
        self.original_graph=e.model_runner.graphs.execute if e.model_runner.graphs else None
        model.recurrent_prepared=self.core
        e.model_runner.runtime.forward=self.forward
        if self.original_graph:
            e.model_runner.graphs.execute=self.graph
        if forced_prefixes is not None:
            e.model_runner.sampler.sample=self.sample
        self.capture_attention=capture_attention
        if capture_attention:
            e.cache_manager._attend_prepared=self.attend
        for i in layers:
            layer=model.model.layers[i]
            for name,module in [('input_norm',layer.input_layernorm),('Q_projection',layer.self_attn.q_proj),
                    ('K_projection',layer.self_attn.k_proj),('V_projection',layer.self_attn.v_proj),
                    ('attention',layer.self_attn),('post_attention_norm',layer.post_attention_layernorm),
                    ('MLP',layer.mlp),('layer_output',layer)]:
                def hook(mod,args,out,layer_idx=i,kind=name):
                    if self.context is not None:
                        meta,index=self.context
                        self.save(f"t{self.token_index}_d{meta['depth']}_l{layer_idx}_{kind}",out[index])
                        if self.token_index==0 and layer_idx==0:
                            self.save(f"t0_d{meta['depth']}_l0_{kind}_all_target_rows",out[meta['target_indices']])
                        if self.token_index==0 and layer_idx<2 and self.capture_attention:
                            positions=meta['target_positions']
                            self.save(f"t0_d{meta['depth']}_l{layer_idx}_{kind}_p{positions[0]}_{positions[-1]}",out[meta['target_indices']])
                self.handles.append(module.register_forward_hook(hook))

    def save(self,key,tensor):
        if self.token_index <= self.stop_index:
            self.tensors[key]=tensor.detach().clone()

    def row(self,metadata):
        if not hasattr(metadata,'rows') or self.token_index>self.stop_index:
            return None
        owners={id(allocation):rid for rid,allocation in metadata.allocations}
        indices=[i for i,(a,d,p) in enumerate(metadata.rows) if owners.get(id(a))==self.rid]
        if not indices:
            return None
        index=indices[-1]
        allocation,depth,pos=metadata.rows[index]
        # Prompt rows are captured at their last actual row only. Decode rows have max_seqlen_q=1.
        return dict(depth=depth,position=pos,owners=[owners.get(id(a)) for a,d,p in metadata.rows],
            target_indices=indices,target_positions=[metadata.rows[i][2] for i in indices],
            actual_rows=len(metadata.rows),packed=metadata.cu_seqlens_q is not None,
            max_q=metadata.max_seqlen_q,allocation_identity=id(allocation),
            physical_pages=list(allocation.block_tables[depth]),
            prefix_frontiers=[w.prefix for w in allocation.written[depth]]),index

    def core(self,hidden,metadata,cache,**kwargs):
        info=self.row(metadata)
        previous=self.context
        self.context=info
        if info:
            meta,index=info
            self.records.append(dict(kind='EAGER_CORE',token_index=self.token_index,**meta,input_rows=len(hidden)))
            self.save(f"t{self.token_index}_d{meta['depth']}_core_input",hidden[index])
        result=self.original_core(hidden,metadata,cache,**kwargs)
        if info:
            self.save(f"t{self.token_index}_d{meta['depth']}_core_output",result[0][index])
        self.context=previous
        return result

    def graph(self,call,hidden):
        info=self.row(call.metadata)
        if info:
            meta,index=info
            actual=hidden if call.entry is None else call.entry.hidden
            self.records.append(dict(kind='GRAPH_CALL' if call.entry is not None else 'EAGER_FALLBACK',
                token_index=self.token_index,**meta,input_rows=len(actual)))
            self.save(f"t{self.token_index}_d{meta['depth']}_core_input",actual[index])
        result=self.original_graph(call,hidden)
        if info:
            self.save(f"t{self.token_index}_d{meta['depth']}_core_output",result[0][index])
        return result

    def forward(self,batch):
        result=self.original_forward(batch)
        # Coda request-row ordering is the scheduled ordering retained by the executor.
        if batch.operation.value=='coda':
            scheduled=self.e.last_schedule
            ids=[r.request.request_id for r in scheduled.requests]
            self.coda_ids=ids
            self.sample_index=0
            if self.rid in ids:
                i=ids.index(self.rid)
                self.save(f't{self.token_index}_coda_hidden',batch.hidden[i])
                self.save(f't{self.token_index}_logits',result.logits[i])
                if self.token_index in self.kv_steps:
                    # Runner stream already waits for this request's producer event.
                    cache=self.e.cache_manager
                    allocation=cache._get_allocation(self.rid)
                    for depth in range(4):
                        for layer in range(24):
                            length=allocation.written[depth][layer].prefix
                            k,v=cache.read(layer,self.rid,depth,length)
                            self.save(f't{self.token_index}_KV_d{depth}_l{layer}_K',k)
                            self.save(f't{self.token_index}_KV_d{depth}_l{layer}_V',v)
                self.token_index+=1
        return result

    def sample(self,logits,params,generator=None):
        native=self.original_sample(logits,params,generator)
        rid=self.coda_ids[self.sample_index]
        self.sample_index+=1
        index=self.sample_counts.get(rid,0)
        self.sample_counts[rid]=index+1
        prefix=self.forced_prefixes.get(rid,[])
        cutoff=self.stop_index if rid==self.rid else len(prefix)
        if index<min(cutoff,len(prefix)):
            # Diagnostic-only common prefix. Never used for serving or quality.
            return native.new_tensor(prefix[index])
        return native

    def attend(self,layer,metadata,q):
        info=self.row(metadata)
        tag=None
        if info and self.token_index==0 and layer<2:
            meta,index=info
            positions=meta['target_positions']
            tag=f"t0_d{meta['depth']}_l{layer}_p{positions[0]}_{positions[-1]}"
            self.save(tag+'_post_rope_Q',q[meta['target_indices']])
            self.save(tag+'_device_positions',metadata.position_ids[meta['target_indices']])
            self.save(tag+'_context_lengths',metadata.context_lengths)
            if metadata.cu_seqlens_q is not None:
                self.save(tag+'_cu_seqlens_q',metadata.cu_seqlens_q)
            k,v=self.e.cache_manager.read(layer,self.rid,meta['depth'],positions[-1]+1)
            self.save(tag+'_prefix_K',k)
            self.save(tag+'_prefix_V',v)
        result=self.original_attend(layer,metadata,q)
        if tag:
            self.save(tag+'_attention_pre_output_projection',result[meta['target_indices']])
        return result

    def finish(self,target,mode):
        self.e.model_runner.synchronize()
        cpu={k:t.cpu() for k,t in self.tensors.items()}
        torch.save(cpu,target/f'{mode}_selected_tensors.pt')
        hashes=[]
        for key,t in cpu.items():
            raw=t.contiguous().view(torch.uint8).numpy().tobytes()
            hashes.append(dict(key=key,shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(raw).hexdigest()))
        write(target/f'{mode}_tensor_index.json',hashes)
        write(target/f'{mode}_core_metadata.json',self.records)
        self.model.recurrent_prepared=self.original_core
        self.e.model_runner.runtime.forward=self.original_forward
        self.e.model_runner.sampler.sample=self.original_sample
        self.e.cache_manager._attend_prepared=self.original_attend
        if self.original_graph:
            self.e.model_runner.graphs.execute=self.original_graph
        for handle in self.handles:
            handle.remove()
        self.context=None
        self.tensors.clear()
        return cpu

def compare(a,b):
    result=[]
    for key in sorted(set(a)&set(b)):
        x,y=a[key],b[key]
        if x.shape!=y.shape:
            result.append(dict(key=key,status='SHAPE_MISMATCH',reference=list(x.shape),candidate=list(y.shape)))
            continue
        exact=torch.equal(x,y)
        if exact:
            error=maximum=0.0
        else:
            delta=x.float()-y.float()
            error=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(x.float()).clamp_min(1e-12))
            maximum=float(delta.abs().max())
        row=dict(key=key,status='COMPARED',exact=exact,relative_L2=error,max_abs=maximum)
        if key.endswith('logits'):
            row.update(reference_argmax=int(x.argmax()),candidate_argmax=int(y.argmax()),
                reference_margin=float(torch.topk(x.float(),2).values.diff().abs()[0]),
                candidate_margin=float(torch.topk(y.float(),2).values.diff().abs()[0]))
        result.append(row)
    return result

@torch.inference_mode()
def run_job(model,job,target):
    from .worker import engine,dispose,bind_legacy
    from poc_flashvep.ouro_chunk_loop_prefill.bench import run
    bind_legacy(job)
    tensors={}
    outputs={}
    prefixes=None
    if job.get('prefix_reference_path'):
        source=__import__('pathlib').Path(job['prefix_reference_path'])
        assert digest(source)==job['prefix_reference_sha256'],'Frozen prefix evidence changed'
        prefixes={r['id']:r['output_ids'] for r in json.loads(source.read_text())['requests']}
    for mode in job.get('modes',['atomic','grouped']):
        e=engine(model,mode,job.get('chunk',1024),8,job)
        run(model,mode,1024,8192,8,0,False,job.get('graphs',True),False,e)
        capture=Capture(model,e,job.get('target','bg_11_3'),stop_index=job.get('stop_index',8),
            layers=job.get('layers',[0,5,11,17,23]),kv_steps=job.get('kv_steps',[0,8]),forced_prefixes=prefixes,
            capture_attention=job.get('capture_attention',False))
        data=run(model,mode,1024,8192,8,job.get('trial',11),False,job.get('graphs',True),True,e)
        write(target/f'{mode}_diagnostic_generation.json',data)
        tensors[mode]=capture.finish(target,mode)
        outputs[mode]={r['id']:r['output_ids'] for r in data['requests']}
        del capture
        dispose(e)
        del e,data
        gc.collect()
        torch.cuda.empty_cache()
    reference=job.get('modes',['atomic','grouped'])[0]
    for mode in outputs:
        if mode!=reference:
            write(target/f'{reference}_vs_{mode}_tensor_comparison.json',compare(tensors[reference],tensors[mode]))
    parity=[dict(id=rid,exact=outputs[reference][rid]==outputs[mode][rid],candidate=mode,
        first_divergent_token=next((i for i,(a,b) in enumerate(zip(outputs[reference][rid],outputs[mode][rid])) if a!=b),None))
        for mode in outputs if mode!=reference for rid in outputs[reference]]
    write(target/'parity.json',parity)
    return dict(scope='HEAVY_DIAGNOSTIC_NOT_CLEAN_TIMING',parity=parity,
        forced_common_prefix=prefixes is not None,
        limitations='Targeted clones can change host timing. Graph replay does not run Python operator hooks. Eager operator controls are separate.')
