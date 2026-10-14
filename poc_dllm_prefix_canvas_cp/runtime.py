"""Native dInfer initialization; no attention, cache or model forward rewrite."""
import ast
import inspect
import os
from pathlib import Path
import torch
from poc_dllm_prefix_canvas_cp.contract import *


def context_settings(prefix):
    assert prefix in PREFIXES and prefix+OUTPUT<=MAX_CONTEXT
    return dict(logical_context=prefix+OUTPUT,physical_key_capacity=2*prefix)


def setup(rank,port,args):
    import poc_dllm_ep_comm.scripts.run_dllm_ep4 as prior
    # Reuse validated distributed/model loading, replacing only configuration
    # constants. Stop before its old cohort and 4096-token runner are created.
    tree=ast.parse(inspect.getsource(prior.worker));function=tree.body[0]
    cut=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='frozen' for t in n.targets))
    function.name='native_setup';function.body=function.body[:cut]+[ast.Return(ast.Call(ast.Name('locals',ast.Load()),[],[]))]
    class ConfigOnly(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='initialize_model_parallel':node.args[1]=ast.Constant(1 if args.backend=='tp4' else 4)
            if isinstance(node.func,ast.Name) and node.func.id=='ServerArgs':
                for kw in node.keywords:
                    if kw.arg=='ep_size':kw.value=ast.Constant(1 if args.backend=='tp4' else 4)
                    if kw.arg=='moe_a2a_backend':kw.value=ast.Constant('none' if args.backend=='tp4' else 'deepep')
            if isinstance(node.func,ast.Name) and node.func.id=='write' and node.args and isinstance(node.args[1],ast.Call):
                # Remove the old EP-only ownership record; write the actual one below.
                if any(k.arg=='topology' for k in node.args[1].keywords):return ast.Constant(None)
            return node
        def visit_Assert(self,node):
            if args.backend=='tp4' and ('owned==' in ast.unparse(node).replace(' ', '') or 'len(ownership)' in ast.unparse(node)):
                return ast.Pass()
            return node
    tree=ConfigOnly().visit(tree);ast.fix_missing_locations(tree)
    prior.MASK=MASK;prior.GPUS=GPUS;prior.UUIDS=UUIDS
    exec(compile(tree,inspect.getfile(prior)+'[configuration-only setup]','exec'),prior.__dict__)
    state=prior.native_setup(rank,port,args)
    model,server=state['model'],state['server']
    from dinfer.decoding.diffusion_runner import ModelRunner
    from dinfer import BlockDiffusionLLM,ThresholdParallelDecoder,BlockIteratorFactory,KVCacheFactory
    graph_probe=bool(getattr(args,'graphs',False));discovery=bool(getattr(args,'discovery',False))
    def make_runner(prefix=None):
        assert not torch.is_grad_enabled(),'Native inference graphs must not capture autograd state'
        # Match native benchmark sizing to this workload. A 32K static buffer
        # for P512 would introduce avoidable native replay_prepare zeroing.
        limit=context_settings(prefix)['logical_context'] if prefix else (1024 if graph_probe else MAX_CONTEXT)
        result=ModelRunner(model,device=str(model.device),server_args=server,max_length=limit,
            enable_cuda_graph=bool(prefix) or graph_probe,enable_compile=False,supported_batch_sizes=[1],block_length=32,
            cache_lengths=[2*prefix] if prefix else ([128,256,512,1024] if graph_probe else None),
            decoding_lengths=[32,64],
            prefill_lengths=[prefix] if prefix and not getattr(args,'eager_prefill',False) else [32],
            use_cross_block=True)
        result._poc_logical_context=limit
        return result
    def make_dllm(runner):
        return BlockDiffusionLLM(runner,ThresholdParallelDecoder(temperature=0,threshold=.9,mask_id=156895,eos_id=156892),
            BlockIteratorFactory(start_block_align=True,use_block_diffusion=True),
            KVCacheFactory('prefix',is_bd_model=True,backend='sglang',max_length=runner._poc_logical_context),
            early_stop=True,maximum_unroll=1,expected_tpf=15,backend='sglang',mini_batch_size=1,
            prefilling_limit=MAX_CONTEXT,use_naive_batching=True)
    runner=make_runner();dllm=make_dllm(runner)
    def make_pair(prefix):
        from poc_dllm_prefix_canvas_cp.graph_events import GraphEvents
        clean=make_runner(prefix)
        bank=GraphEvents();bank.install(model)
        try:marked=make_runner(prefix)
        finally:bank.restore()
        clean._poc_marked=(marked,make_dllm(marked),bank)
        return clean,make_dllm(clean)
    if discovery:runner._poc_factory=make_pair
    layers=[]
    for l in model.model.layers:
        a=l.attention
        layers.append(dict(layer=l.layer_id,q_heads=a.num_heads,kv_heads=a.num_kv_heads,head_dim=a.head_dim,
            qkv_shape=list(a.query_key_value.weight.shape),
            mlp_weights={n:list(v.shape) for n,v in l.mlp.named_parameters()},
            layer_scatter=str(l.layer_communicator.layer_scatter_modes)))
    write(Path(args.output)/f'actual_layout.rank{rank}.json',dict(rank=rank,physical_gpu=GPUS[rank],
        cuda_uuid=str(torch.cuda.get_device_properties(rank).uuid),backend=args.backend,
        TP=4,EP=1 if args.backend=='tp4' else 4,CP=1,DP=1,PP=1,layers=layers,
        KV_ownership='head-sharded: one of four KV heads/rank, complete sequence on each owner',
        max_position=model.config.max_position_embeddings,runner_max_length=runner.max_length,
        graph_enabled=runner.enable_cuda_graph,compiled=runner.enable_compile,
        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))
    return model,runner,dllm,state['tokenizer']
