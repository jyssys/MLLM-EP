"""Bounded FA3 address compatibility microbenchmark. No model engine change."""
import datetime
import statistics
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch
from flash_attn_interface import flash_attn_with_kvcache

@torch.inference_mode()
def measure(fn):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):value=fn()
    graph.replay();torch.cuda.synchronize()
    samples=[]
    for _ in range(31):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize()
        samples.append(a.elapsed_time(b)/16)
    return dict(samples_ms=samples,median_ms=statistics.median(samples),inner=16,
        boundary='Local GPU stream events across repeated complete attention calls.'),value

@torch.inference_mode()
def main():
    mask=validate_mask();assert len(mask)==1
    torch.cuda.set_device(0)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[GPUS.index(mask[0])]
    out=Path(os.environ['LOOPED_KV_OUTPUT']);records=[]
    # These are deterministic API fixtures, not random model weights or a
    # pretrained task-quality result. Every depth has a different K/V payload.
    for N in [1024,4096,8192]:
        for B in [1,2,4,8]:
            lengths=torch.tensor([N-16*(i%3) for i in range(B)],device='cuda',dtype=torch.int32)
            depths=torch.tensor([i%4 for i in range(B)],device='cuda',dtype=torch.int64)
            raw=torch.arange(B*4*N*16*128,device='cuda',dtype=torch.int64).reshape(B,4,N,16,128)
            k=(torch.sin(raw.float()*.013)*.5).to(torch.bfloat16)
            v=(torch.cos(raw.float()*.007)*.5).to(torch.bfloat16)
            q=(torch.sin(torch.arange(B*16*128,device='cuda',dtype=torch.float32)*.019)*.5).to(torch.bfloat16).reshape(B,1,16,128)
            indices=torch.arange(B,device='cuda')
            selected_k=k[indices,depths].contiguous();selected_v=v[indices,depths].contiguous()
            def contiguous():return flash_attn_with_kvcache(q,selected_k,selected_v,cache_seqlens=lengths,causal=True)
            ref_timing,reference=measure(contiguous)
            for page in [16,1]:
                pages=N//page
                arena_k=k.reshape(B*4*pages,page,16,128)
                arena_v=v.reshape_as(arena_k)
                table=((indices*4+depths)[:,None]*pages+torch.arange(pages,device='cuda')[None,:]).to(torch.int32)
                def paged():return flash_attn_with_kvcache(q,arena_k,arena_v,cache_seqlens=lengths,page_table=table,causal=True)
                timing,result=measure(paged)
                diff=(result-reference).float()
                # Fixed before results. This is kernel API numerical tolerance,
                # not a declaration of full-model generation parity.
                close=torch.allclose(result,reference,atol=.005,rtol=.005)
                assert close,(N,B,page,float(diff.abs().max()))
                append_check='NOT_RUN'
                if N==1024 and B==4:
                    old_k=k[:,:,N-1].clone();old_v=v[:,:,N-1].clone()
                    new_k=(q*.75).contiguous();new_v=(q*.25).contiguous()
                    append_lengths=torch.full((B,),N-1,device='cuda',dtype=torch.int32)
                    ref_k=selected_k.clone();ref_v=selected_v.clone()
                    append_ref=flash_attn_with_kvcache(q,ref_k,ref_v,k=new_k,v=new_v,
                        cache_seqlens=append_lengths,causal=True)
                    append_result=flash_attn_with_kvcache(q,arena_k,arena_v,k=new_k,v=new_v,
                        cache_seqlens=append_lengths,page_table=table,causal=True)
                    torch.cuda.synchronize()
                    assert torch.allclose(append_result,append_ref,atol=.005,rtol=.005)
                    expected_k=old_k.clone();expected_v=old_v.clone()
                    expected_k[indices,depths]=new_k[:,0];expected_v[indices,depths]=new_v[:,0]
                    assert torch.equal(k[:,:,N-1],expected_k)
                    assert torch.equal(v[:,:,N-1],expected_v)
                    # Restore the fixture before the next page-layout comparison.
                    k[:,:,N-1].copy_(old_k);v[:,:,N-1].copy_(old_v)
                    append_check='PASS unique private depth rows and unchanged other depths'
                records.append(dict(context=N,B=B,page_size=page,physical_gpu=mask[0],
                    contiguous=ref_timing,paged=timing,atol=.005,rtol=.005,
                    max_abs=float(diff.abs().max()),relative_L2=float(diff.norm()/reference.float().norm()),
                    numerical_PASS=bool(close),table_bytes=table.numel()*table.element_size(),
                    speedup_contiguous_over_paged=ref_timing['median_ms']/timing['median_ms'],
                    source_depths=depths.tolist(),append_check=append_check,label='GPU_MICROBENCH'))
                write(out/'paged_depth.json',dict(records=records,
                    fixture='Deterministic distinct-depth BF16 K/V. No model-weight or task-quality claim.',
                    scope='FA3 address support and bounded private append checks. Compact allocator, terminal aliases, original gate, and live generation remain NOT_RUN.',
                    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    implementation_sha256=sha(SHARED_DEPS/'python_cuda129/flash_attn_interface.py')))
            print('PAGED_DEPTH',N,B,flush=True)
            del raw,k,v,arena_k,arena_v,selected_k,selected_v
    write(out/'status.json',dict(success=True,cases=len(records),quality='NOT_RUN',live_architecture='NOT_RUN'))

if __name__=='__main__':main()
