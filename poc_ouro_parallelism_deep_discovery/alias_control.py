"""Process-local weight-storage counterfactual. No checkpoint file changes."""
import copy
import hashlib
import inspect
from pathlib import Path
import types

def storage_forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
    hidden_states=inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    for current_ut in range(self.total_ut_steps):
        residual=None
        for layer in self._campaign_execution[current_ut]:
            hidden_states,residual=layer(positions,hidden_states,current_ut,residual)
        hidden_states,_=self.norm(hidden_states,residual)
    return hidden_states

def clone_module(module,shared_attention_ids):
    import torch
    from torch import nn
    if id(module) in shared_attention_ids:return module
    result=copy.copy(module)
    result._parameters={name:None if value is None else nn.Parameter(value.detach().clone(),requires_grad=False)
                        for name,value in module._parameters.items()}
    result._buffers=dict(module._buffers)
    result._modules={name:None if value is None else clone_module(value,shared_attention_ids)
                     for name,value in module._modules.items()}
    # CustomOp stores _forward_method as an instance-bound method. A shallow
    # copy must not retain the old module as that method's self.
    for name,value in vars(module).items():
        if isinstance(value,types.MethodType) and value.__self__ is module:
            setattr(result,name,types.MethodType(value.__func__,result))
    return result

def install(mode):
    assert mode in ['alias','copies']
    import torch
    from torch import nn
    from vllm.model_executor.models import ouro
    assert not getattr(ouro.OuroForCausalLM,'_campaign_storage_installed',False)
    ordinary_load=ouro.OuroForCausalLM.load_weights

    def load_weights(self,weights):
        loaded=ordinary_load(self,weights)
        names_before={name for name,_ in self.named_parameters()}
        core=self.model;layers=core.layers;R=core.total_ut_steps
        assert R==4 and len(layers) in [24,48]
        assert core.start_layer==0 and core.end_layer==len(layers),'This counterfactual supports PP1 only.'
        copies=[layers]
        for _ in range(3):
            cloned=[]
            for layer in layers:
                assert not list(layer.self_attn.rotary_emb.parameters())
                shared={id(a) for a in layer.self_attn.attn}|{id(layer.self_attn.rotary_emb)}
                cloned.append(clone_module(layer,shared))
            copies.append(nn.ModuleList(cloned))
        comparisons=[]
        for depth in range(1,4):
            count=0;amount=0
            for index,(original,replica) in enumerate(zip(layers,copies[depth])):
                assert all(a is b for a,b in zip(original.self_attn.attn,replica.self_attn.attn))
                assert original.self_attn.rotary_emb is replica.self_attn.rotary_emb
                source=dict(original.named_parameters());target=dict(replica.named_parameters())
                assert source.keys()==target.keys()
                for name,value in source.items():
                    other=target[name]
                    assert value.dtype==other.dtype==torch.bfloat16,(index,name,value.dtype,other.dtype)
                    assert value.untyped_storage().data_ptr()!=other.untyped_storage().data_ptr()
                    assert torch.equal(value.contiguous().view(torch.int16),other.contiguous().view(torch.int16)),(depth,index,name)
                    count+=1;amount+=value.numel()*value.element_size()
            comparisons.append(dict(depth=depth,parameter_tensors=count,parameter_bytes=amount,
                all_values_bitwise_equal=True,all_parameter_storages_distinct=True,
                attention_objects_shared=True))
        core._campaign_physical_copies=nn.ModuleList(copies)
        # Every arm keeps all copies allocated. Only the execution aliases differ.
        core._campaign_execution=nn.ModuleList([layers for _ in range(4)] if mode=='alias' else copies)
        core._campaign_storage_evidence=dict(mode=mode,comparisons=comparisons,
            checkpoint_layer_identities=len(layers),registered_parameter_block_copies=4*len(layers),logical_depth=4*len(layers),
            copies_allocated=4,execution_storage='one shared core' if mode=='alias' else 'four distinct value-identical cores',
            precision='BF16',attention_contract='Original R4 attention objects and static forward-context entries.',
            rotary_contract='Original parameter-free RoPE module remains shared, including its cache dtype update.',
            forward_source=inspect.getsource(storage_forward),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            allocated_bytes_after_setup=torch.cuda.memory_allocated(),reserved_bytes_after_setup=torch.cuda.memory_reserved(),
            quality='Four-position numerical check and controlled output only. Natural quality NOT_RUN.')
        # The native loader tracks initialized names after this method returns.
        # Preserve its check on original names. Only explicitly initialized
        # copies add new initialized names.
        copied_names={name for name,_ in self.named_parameters()}-names_before
        assert all(name.startswith('model._campaign_physical_copies.') for name in copied_names)
        core._campaign_storage_evidence['explicitly_initialized_copy_names']=sorted(copied_names)
        core._campaign_storage_evidence['original_loaded_name_count']=len(loaded)
        return loaded|copied_names

    ouro.OuroModel.forward=storage_forward
    ouro.OuroForCausalLM.load_weights=load_weights
    ouro.OuroForCausalLM._campaign_storage_installed=True
    return inspect.getsource(storage_forward)
