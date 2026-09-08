"""Temporary-port diagnostic: locate first HF/native hidden-state divergence.

Only explicitly selected diagnostic forwards capture tensors. Not performance
evidence and never enabled in the clean EP timing harness.
"""
SELECTED=(0,1,2,3,4,12,24,47)


def install_hidden_diagnostic(layers, *, native=False, rank=0, local_m=None):
    state={"active":False,"records":{}}
    def save(name,value):
        if not state["active"]:return
        if isinstance(value,tuple):value=value[0]
        value=value.reshape(-1,value.shape[-1])
        if native and len(value)!=local_m:
            assert len(value)==4*local_m,(name,value.shape,local_m)
            value=value.chunk(4,dim=0)[rank]
        state["records"][name]=value.detach().clone()
    for i,layer in enumerate(layers):
        if i not in SELECTED:continue
        def pre(module,args,kwargs,i=i):
            if not state["active"]:return
            hidden=kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            residual=kwargs.get("residual") if native else None
            if residual is not None:hidden=hidden+residual
            save(f"layer{i}_input",hidden)
        layer.register_forward_pre_hook(pre,with_kwargs=True)
        def attention(module,args,output,i=i):save(f"layer{i}_attention_output",output)
        layer.self_attn.register_forward_hook(attention)
        def router(module,args,i=i):save(f"layer{i}_router_input",args[0])
        layer.mlp.gate.register_forward_pre_hook(router)
    return state
