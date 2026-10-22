"""Pretrained G1/G2/G3/G4 numerical tests. Not model quality evaluation."""
import gc
import json
import torch
from .common import ROOT, MODEL, write, config

@torch.inference_mode()
def run_job(model,job,target):
    from transformers import AutoTokenizer
    from vllm_rlt import SamplingParams
    from vllm_rlt.request import RequestPhase
    from .worker import engine,dispose
    from poc_flashvep.ouro_chunk_loop_prefill.smoke import differences as full_differences
    def differences(a,b):
        # Exact equality implies both reported error values are zero.
        # Avoid redundant FP32 conversion of multi-GiB equal CPU KV snapshots.
        if torch.equal(a,b):
            return dict(exact=True,max_abs=0.0,rel_l2=0.0)
        return full_differences(a,b)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    # Explicit numerical fixture, not a natural-serving workload.
    ids=tokenizer.encode('A rectangle has length twelve and width seven. Its area equals eighty-four. Explain the multiplication. '*3000)
    all_rows=[]
    for length in job.get('lengths',[128,512,2048,8192]):
        reference=None
        for mode in job.get('modes',['atomic','split','grouped','grouped3','staged4','mixed']):
            e=engine(model,mode,job['chunk'],2,dict(job,blocks=max(1024,4*((length+48+15)//16))))
            e.add_request('probe',ids[:length],SamplingParams(max_tokens=8,max_loops=4,exit_threshold=1.0,temperature=0))
            hidden=logits=None
            kv=[]
            spans=[]
            while e.has_unfinished_requests():
                request=e.scheduler.requests.get('probe')
                if request and request.phase==RequestPhase.CODA and hidden is None:
                    e.model_runner.synchronize()
                    hidden=request.hidden_state.clone().cpu()
                    logits=model.coda(request.hidden_state.reshape(1,-1)).cpu()
                    for depth in range(4):
                        for layer in range(24):
                            k,v=e.cache_manager.read(layer,'probe',depth,length)
                            kv.extend((k.cpu(),v.cpu()))
                outputs=e.step()
                for r in e.last_schedule.requests:
                    phase=e.last_schedule.phase_for(r).value
                    if phase in ('prefill','prefilling_recurrent'):
                        group=4 if phase=='prefill' else r.loop_count
                        spans.append(dict(start=r.token_start,tokens=r.token_count,depth=r.depth,group=group))
                for output in outputs:
                    final=output
            e.model_runner.synchronize()
            row=dict(mode=mode,P=length,C=job['chunk'],graphs=job.get('graphs',True),streams=job.get('streams',True),
                tokens=list(final.token_ids),depths=list(final.exit_depths),prefill_spans=spans,
                exact_prompt_work=sum(s['tokens']*s['group'] for s in spans)==4*length)
            assert row['exact_prompt_work'],row
            assert all(d==4 for d in row['depths']),row
            if reference is None:
                reference=(row,hidden,logits,kv)
            else:
                a,h,l,v=reference
                row['comparison']=dict(token_exact=a['tokens']==row['tokens'],depth_exact=a['depths']==row['depths'],
                    hidden=differences(h,hidden),logits=differences(l,logits),KV_all_exact=all(torch.equal(x,y) for x,y in zip(v,kv)),
                    KV_max_relative_L2=max(differences(x,y)['rel_l2'] for x,y in zip(v,kv)))
            write(target/f'P{length}_{mode}.json',row)
            all_rows.append(row)
            dispose(e)
            del e,hidden,logits,kv
            gc.collect()
            torch.cuda.empty_cache()
        del reference
        gc.collect()
    passes=all(r['comparison']['token_exact'] and r['comparison']['KV_all_exact']
        and r['comparison']['hidden']['exact'] and r['comparison']['logits']['exact'] for r in all_rows if 'comparison' in r)
    return dict(isolated_exact_pass=passes,comparisons=sum('comparison' in r for r in all_rows),
        scope='ISOLATED_PRETRAINED_NUMERICAL_NOT_FREE_RUNNING_BENCHMARK_QUALITY')
