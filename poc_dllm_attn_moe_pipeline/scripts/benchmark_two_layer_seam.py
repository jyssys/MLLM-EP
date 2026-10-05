"""Stage-A producer split tax and native consumer component measurements.

Not a two-layer overlap result. Full seam arms remain gated/NOT_RUN until A.
All expert/combine/shared/source-layout work is actual four-GPU execution.
"""
import ast
import inspect
import types
import torch
import torch.distributed as dist
from poc_dllm_attn_moe_pipeline.contract import write
from poc_dllm_attn_moe_pipeline.attention import partial_state, merge_states


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def measure(fn, warmups=3, repeats=5):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    rows = []
    for n in range(repeats):
        dist.barrier()
        a = event()
        output = fn()
        b = event()
        torch.cuda.synchronize()
        rows.append(a.elapsed_time(b)*1000)
    return rows, output


def shared_overlap_function(block):
    """Move the same native shared operation, not change arithmetic/scaling."""
    import dinfer.model.modeling_llada2_moe_sglang as native
    tree = ast.parse(inspect.getsource(native.LLaDA2SparseMoeBlock.forward_deepep).lstrip())
    function = tree.body[0]
    function.name = 'forward_shared_overlap'
    router_index = next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=='router_logits' for t in n.targets))
    setup = ast.parse('''
current_stream = torch.cuda.current_stream()
self._seam_shared_stream.wait_stream(current_stream)
with torch.cuda.stream(self._seam_shared_stream):
    local_shared = self.shared_experts(local_hidden)
''').body
    function.body[router_index:router_index] = setup
    for n in function.body:
        if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Attribute) and n.test.left.attr=='num_shared_experts':
            # Original branch contains shared compute then routed+shared add.
            n.body[0] = ast.parse('current_stream.wait_stream(self._seam_shared_stream)').body[0]
    ast.fix_missing_locations(tree)
    namespace = native.__dict__.copy()
    exec(compile(tree,inspect.getfile(native)+'[shared-only-adaptation]','exec'),namespace)
    block._seam_shared_stream = torch.cuda.Stream()
    return types.MethodType(namespace['forward_shared_overlap'],block)


def benchmark_components(model,snapshots,rank,out):
    records = []
    for lid,snap in snapshots.items():
        if rank == 0:
            print('COMPONENT_BOUNDARY',lid,'BEGIN',flush=True)
        block = model.model.layers[lid].mlp
        x = snap['moe_input']
        assert x.shape == (1,32,2048)
        def unsplit():
            return block.forward_deepep(x,False)
        def split():
            # Same global positions/source ownership: each source's8 rows split4+4.
            # Group0 is [0:4,8:12,16:20,24:28], not naive contiguous16 ownership.
            y = torch.empty_like(x)
            release = []
            for g in range(2):
                group = torch.cat([x[:,r*8+g*4:r*8+g*4+4] for r in range(4)],dim=1)
                result = block.forward_deepep(group,False)
                for r in range(4):
                    y[:,r*8+g*4:r*8+g*4+4] = result[:,r*4:r*4+4]
                release.append(event())
            return y,release
        # Counterbalanced paired producer repetitions, not select fastest runs.
        for _ in range(3):
            unsplit()
            split()
        torch.cuda.synchronize()
        reference = unsplit()
        torch.cuda.synchronize()
        for rep in range(5):
            values = {}
            for variant in (('unsplit','split_serial') if rep%2==0 else ('split_serial','unsplit')):
                dist.barrier()
                a = event()
                output = unsplit() if variant=='unsplit' else split()
                b = event()
                torch.cuda.synchronize()
                y = output if variant=='unsplit' else output[0]
                values[variant] = a.elapsed_time(b)*1000
                diff = y.float()-reference.float()
                entry = dict(kind='producer',layer=lid,variant=variant,repetition=rep,
                             duration_us=values[variant],max_abs=float(diff.abs().max()),
                             relative_l2=float(diff.norm()/reference.float().norm()),
                             groups=1 if variant=='unsplit' else 2,source_rows=8,
                             includes='router dispatch expert combine eventwait shared add source_all_gather repack',
                             tp_visible_gap_us=None if variant=='unsplit' else output[1][0].elapsed_time(output[1][1])*1000)
                records.append(entry)
        # Strong generic shared-expert overlap separately, no prefix method gain.
        shared_fn = shared_overlap_function(block)
        times,shared_result = measure(lambda:shared_fn(x,False))
        diff = shared_result.float()-reference.float()
        records.append(dict(kind='generic_shared',layer=lid,times_us=times,
                            max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/reference.float().norm())))
        write(out/f'components.rank{rank}.json',dict(rank=rank,records=records,status='PARTIAL_APPEND_PROGRESS'))
        if rank == 0:
            print('COMPONENT_BOUNDARY',lid,'PRODUCER_DONE',flush=True)
        # Host-variable DeepEP receive counts can prevent graph capture.
        # Pinned ServerArgs explicitly disables graph for DeepEP normal mode.
        # First diagnostic capture stalled in CPU receive-count coordination;
        # preserve failure log. Do not repeat an unsupported distributed capture.
        graph_status = 'UNSUPPORTED: installed ServerArgs disables CUDA graph with deepep_mode=normal; bounded initial capture aborted'
        records.append(dict(kind='graph_capture',layer=lid,status=graph_status))
        c = snap.get('consumer')
        if c is None:
            records.append(dict(kind='consumer',layer=lid,status='NO_CAPTURE'))
            continue
        next_layer = model.model.layers[lid+1]
        attn = next_layer.attention
        h,r = c['hidden_states'],c['residual']
        def qkv_work():
            normalized,_ = next_layer.layer_communicator.prepare_attn(h.clone(),r.clone(),None)
            qkv,_ = attn.query_key_value(normalized)
            q,k,v = qkv.split([attn.q_size,attn.kv_size,attn.kv_size],dim=-1)
            q,k = attn._apply_qk_norm(q,k)
            q,k = attn.rotary_emb(c['positions'].flatten(),q.flatten(0,1),k.flatten(0,1),fused_set_kv_buffer_arg=None)
            q = q.view(1,32,attn.num_heads,attn.head_dim).transpose(1,2)
            k = k.view(1,32,attn.num_kv_heads,attn.head_dim).transpose(1,2)
            v = v.view(1,32,attn.num_kv_heads,attn.head_dim).transpose(1,2)
            return q,k,v
        qkv_times,(q,k,v) = measure(qkv_work)
        cached = c['past_key_values']
        mask = c['attention_mask']
        native_mask_none = mask is None
        if mask is None:
            # Preserve native unmasked SDPA exactly; do not invent a new mask.
            mask = torch.ones((1,1,32,cached[0].shape[2]),dtype=torch.bool,device=q.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        prefix_length = cached[0].shape[2]-32
        # Preserve exactly the native mask-permitted earlier-cache slots.
        # When native mask=None, padding remains exposed; do not silently fix it.
        prefix_k,prefix_v = cached[0][:,:,:prefix_length],cached[1][:,:,:prefix_length]
        from dinfer.model.modeling_llada2_moe_sglang import repeat_kv
        prefix_k,prefix_v = repeat_kv(prefix_k,attn.num_key_value_groups),repeat_kv(prefix_v,attn.num_key_value_groups)
        k,v = repeat_kv(k,attn.num_key_value_groups),repeat_kv(v,attn.num_key_value_groups)
        p_mask,c_mask = mask[...,:prefix_length],mask[...,prefix_length:]
        prefix_times,p_state = measure(lambda:partial_state(q,prefix_k,prefix_v,p_mask))
        current_times,c_state = measure(lambda:partial_state(q,k,v,c_mask))
        merge_times,merged = measure(lambda:merge_states(p_state,c_state))
        native = torch.nn.functional.scaled_dot_product_attention(q,torch.cat([prefix_k,k],dim=2),
                torch.cat([prefix_v,v],dim=2),attn_mask=mask,dropout_p=0,is_causal=False)
        independent = merge_states(partial_state(q,torch.cat([prefix_k,k],dim=2),
                                  torch.cat([prefix_v,v],dim=2),mask),partial_state(q,k[:,:,:0],v[:,:,:0]))
        torch.cuda.synchronize()
        diff = merged-native.float()
        records.append(dict(kind='consumer',layer=lid,status='COMPONENT_ONLY_NOT_SEAM',prefix_cache_slots=prefix_length,
                            prefix_valid_tokens=int(p_mask.any(dim=-2).sum()),current_tokens=32,
                            logical_prior_prefix_tokens=int(c['positions'].min()),native_attention_mask_none=native_mask_none,
                            padded_prefix_slots=prefix_length-int(c['positions'].min()),
                            qkv_times_us=qkv_times,prefix_times_us=prefix_times,current_times_us=current_times,
                            merge_times_us=merge_times,max_abs=float(diff.abs().max()),
                            relative_l2=float(diff.norm()/native.float().norm()),
                            fp32_reference_max_abs=float((merged-independent).abs().max()),
                            includes='native residual/norm QKV QK_norm RoPE, FP32 online-softmax reference',
                            cache_current_fresh=True,accepted_current_never_prefix=True))
        if rank == 0:
            print('COMPONENT_BOUNDARY',lid,'CONSUMER_DONE',flush=True)
        write(out/f'components.rank{rank}.json',dict(rank=rank,records=records,
            runtime_overlap_arms='NOT_RUN until StageA gate',synthetic_prefix_sweep='NOT_RUN'))
