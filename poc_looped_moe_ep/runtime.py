"""Surgical native-Foil expert-expression bridge; no attention/cache engine."""
import ast
import inspect
import sys
import textwrap
import torch
import torch.nn.functional as F
import torch.distributed as dist
from poc_looped_moe_ep.contract import *


def grouped_mm(a,b,*,offs):
    return torch._grouped_mm(a,b,offs=offs)


def restoration_buffer(flat):
    return torch.zeros_like(flat,memory_format=torch.contiguous_format)


def load_model(device):
    if not hasattr(F,'grouped_mm'):F.grouped_mm=grouped_mm
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model,info=AutoModelForCausalLM.from_pretrained(MODEL,subfolder=SUBFOLDER,
        revision=REVISION,trust_remote_code=True,torch_dtype=torch.bfloat16,
        output_loading_info=True)
    assert not any(info.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
    model.eval();model.to(device)
    # HF dtype conversion applies to floating buffers too. Restore official
    # FP32 RoPE buffers exactly from the pinned file, rather than rounding them.
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    path=snapshot_download(MODEL,revision=REVISION,local_files_only=True)
    with safe_open(Path(path)/SUBFOLDER/'model.safetensors',framework='pt',device='cpu') as source:
        for name,module in model.named_modules():
            if hasattr(module,'rotations'):
                module.rotations=source.get_tensor(name+'.rotations').to(device)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,subfolder=SUBFOLDER,
        revision=REVISION,trust_remote_code=True)
    return model,tokenizer,path


def pools(model):
    return [('head',model.model.head_layers[0]),('core',model.model.stack.layers[0]),
            ('tail',model.model.tail_layers[0])]


class PhysicalEP:
    def __init__(self,model,rank,core_placement=None):
        from poc_multibd_moe_ep.bounded_bridge import repair_normal_owner_local_ids
        repair_normal_owner_local_ids()
        from diffulex.moe.dispatcher.deepep_dispatcher import DeepEPDispatcher
        self.rank=rank;self.model=model;self.mode='clean';self.records=[];self.forward=0
        self.ownership=[];self.core_weights_cpu={}
        for pool,block in pools(model):
            n=block.num_experts;local=n//4
            placement=(core_placement if pool=='core' and core_placement is not None
                       else [e//local for e in range(n)])
            assert len(placement)==n and all(placement.count(r)==local for r in range(4))
            by_owner=[e for r in range(4) for e in range(n) if placement[e]==r]
            inverse=[by_owner.index(e) for e in range(n)]
            owned=by_owner[rank*local:(rank+1)*local]
            tensors=[]
            for name in ['experts_w1','experts_w2','experts_w3']:
                old=getattr(block,name).data
                if pool=='core':self.core_weights_cpu[name]=old.detach().cpu()
                selected=old[owned].clone()
                assert torch.equal(selected,old[owned])
                setattr(block,name,torch.nn.Parameter(selected,requires_grad=False))
                tensors.append(dict(name=name,shape=list(selected.shape),dtype=str(selected.dtype),
                                    bytes=selected.numel()*selected.element_size(),
                                    samples=selected[:,0,:8].detach().cpu().tolist()))
            block._poc_pool=pool;block._poc_inverse=torch.tensor(inverse,device='cuda',dtype=torch.int64)
            block._poc_runtime=self
            block._poc_dispatcher=DeepEPDispatcher(ep_group=dist.group.WORLD,ep_size=4,
                num_local_experts=local,num_experts=n,top_k=2,hidden_size=1024,
                params_dtype=torch.bfloat16,deepep_mode='normal',async_finish=False)
            self.ownership.append(dict(pool=pool,owned_experts=owned,placement=placement,
                tensors=tensors,expert_storage_pointers=[getattr(block,name).data_ptr() for name in
                                                        ['experts_w1','experts_w2','experts_w3']]))
        torch.cuda.empty_cache()
        block_class=type(model.model.stack.layers[0]);module=sys.modules[block_class.__module__]
        if not hasattr(block_class,'_poc_original_forward'):
            original=block_class.forward
            tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
            body=tree.body[0].body
            start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
                       any(isinstance(t,ast.Name) and t.id=='x_flat' for t in n.targets))
            end=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id=='experts_out' for t in n.targets))
            body[start:end+1]=ast.parse('experts_out, counts = _poc_ep(self, norm2_out, top_experts, top_scores, loop_step)').body
            scope=dict(vars(module));scope['_poc_ep']=lambda block,*args:block._poc_runtime.run(block,*args)
            exec(compile(ast.fix_missing_locations(tree),'<poc-native-expert-expression-only>','exec'),scope)
            block_class._poc_original_forward=original
            block_class.forward=scope['forward']

    def set_core_placement(self,placement):
        """Static-arm setup only, always before warmup/timing; never per loop."""
        assert len(placement)==64 and all(placement.count(r)==16 for r in range(4))
        block=self.model.model.stack.layers[0]
        by_owner=[e for rank in range(4) for e in range(64) if placement[e]==rank]
        owned=by_owner[self.rank*16:(self.rank+1)*16]
        inverse=[by_owner.index(e) for e in range(64)]
        block._poc_inverse=torch.tensor(inverse,device='cuda',dtype=torch.int64)
        info=next(row for row in self.ownership if row['pool']=='core')
        info.update(owned_experts=owned,placement=placement)
        for row in info['tensors']:
            name=row['name'];value=self.core_weights_cpu[name][owned].to('cuda')
            setattr(block,name,torch.nn.Parameter(value,requires_grad=False))
            row['samples']=value[:,0,:8].cpu().tolist()
        info['expert_storage_pointers']=[getattr(block,name).data_ptr() for name in
                                        ['experts_w1','experts_w2','experts_w3']]
        torch.cuda.synchronize()

    def run(self,block,hidden,ids,weights,loop):
        flat=hidden.reshape(-1,hidden.shape[-1]);all_ids=ids.reshape(-1,2)
        all_weights=weights.reshape(-1,2)
        h=flat[self.rank::4].contiguous()
        mapped=block._poc_inverse[all_ids[self.rank::4]].contiguous()
        w=all_weights[self.rank::4].float().contiguous()
        events=[torch.cuda.Event(enable_timing=True) for _ in range(5)] if self.mode!='clean' else None
        if events:events[0].record()
        dispatched=block._poc_dispatcher.dispatch(h,mapped,w)
        if events:events[1].record()
        meta=dispatched.metadata;x=dispatched.recv_hidden_states
        if meta.total_recv_slots:
            offsets=meta.num_recv_tokens_per_expert.cumsum(0).to(torch.int32)
            a=grouped_mm(x,block.experts_w1,offs=offsets)
            b=grouped_mm(x,block.experts_w3,offs=offsets)
            branch=grouped_mm(F.silu(a)*b,block.experts_w2,offs=offsets)
            branch=branch*meta.recv_weights.to(branch.dtype).unsqueeze(-1)
        else:branch=x
        if events:events[2].record()
        local=block._poc_dispatcher.combine(branch,meta)
        if events:events[3].record()
        restored=restoration_buffer(flat)
        restored[self.rank::4]=local
        dist.all_reduce(restored)
        if events:events[4].record()
        if self.mode!='clean':
            record=dict(pool=block._poc_pool,loop=int(loop),forward=self.forward,rank=self.rank,
                rows=h.shape[0],recv_rows=meta.native_recv_num_tokens,recv_paths=meta.total_recv_slots,
                events=events)
            if self.mode=='trace':
                record.update(ids=all_ids[self.rank::4].detach().clone(),
                              weights=all_weights[self.rank::4].detach().clone())
            self.records.append(record)
        counts=torch.bincount(all_ids.flatten(),minlength=block.num_experts)
        return restored.reshape_as(hidden),counts

    def resolved(self):
        torch.cuda.synchronize();result=[]
        for record in self.records:
            row={k:v for k,v in record.items() if k!='events'}
            events=record['events']
            row.update(dispatch_ms=events[0].elapsed_time(events[1]),
                       expert_ms=events[1].elapsed_time(events[2]),
                       combine_ms=events[2].elapsed_time(events[3]),
                       restoration_ms=events[3].elapsed_time(events[4]))
            for key in ['ids','weights']:
                if key in row:row[key]=row[key].cpu().tolist()
            result.append(row)
        self.records=[];return result
