"""Controlled legal alternatives. Forced actions are never the adaptive method."""
import gc
import gzip
import hashlib
import json
import time
from collections import defaultdict
import torch
from .common import ROOT,guard,write
from ..ouro_budget_joint_cg.benchmark import engine
from ..ouro_budget_joint_cg.fairness import attach
from ..ouro_budget_joint_cg.service_observer import ServiceObserver

def fingerprint(e):
    e.model_runner.synchronize()
    result=[]
    for rid,request in sorted(e.scheduler.requests.items()):
        length=len(request.prompt_token_ids)+len(request.generated_token_ids)-1
        assert length>=len(request.prompt_token_ids)
        h=hashlib.sha256()
        for depth in range(4):
            for layer in range(24):
                k,v=e.cache_manager.read(layer,rid,depth,length)
                for value in (k,v):h.update(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        hidden=request.hidden_state
        hidden_hash=None if hidden is None else hashlib.sha256(hidden.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        result.append(dict(id=rid,phase=request.phase.value,issued_depth=request.loops_done,
            output_ids=list(request.generated_token_ids),position=request.position,valid_KV_length=length,
            KV_sha256=h.hexdigest(),hidden_sha256=hidden_hash))
    return result

@torch.inference_mode()
def run_job(model,job,target):
    from vllm_rlt import SamplingParams
    cases={c['id']:c for c in json.loads(gzip.open(ROOT/'workload_payloads.json.gz','rt').read())['cases']}
    lock=json.loads((ROOT/'workload_lock.json').read_text());stops=lock['frontend_stop_ids']
    background=[cases[c] for c in job['background_ids']];long=cases[job['target_id']]
    summaries=[]
    for cell in job['cells']:
        e=engine(model,cell['runtime'],cell['C'],4,job);fairness=attach(e,True)
        params=SamplingParams(max_tokens=128,max_loops=4,exit_threshold=1.,temperature=0.)
        tokens={};clocks=defaultdict(list);finished=set()
        def consume(outputs):
            for output in outputs:
                now=time.perf_counter();rid=output.request_id;ids=list(output.token_ids)
                if len(ids)>len(tokens.get(rid,[])):
                    assert len(ids)==len(tokens.get(rid,[]))+1
                    clocks[rid].append(now);tokens[rid]=ids
                assert all(d==4 for d in output.exit_depths)
                if output.finished:finished.add(rid)
                elif ids and ids[-1] in stops:
                    e.abort_request(rid);finished.add(rid)
        for i,case in enumerate(background):e.add_request(f'b{i}',case['input_ids'],params)
        steps=0
        while e.has_unfinished_requests() and not all(len(tokens.get(f'b{i}',[]))>=16 or f'b{i}' in finished for i in range(3)):
            consume(e.step());steps+=1
            assert steps<2000,'Cannot reach controlled background boundary'
        initial=fingerprint(e)
        observer=ServiceObserver(e,dict(run_id=job['id'],label=cell['label'],physical_GPU=guard(),
            lane='LIVE_MEASURED_FORCED_ALTERNATIVE_COMPONENT_NOT_ADAPTIVE_GAIN'))
        begin=time.perf_counter()
        e.add_request('target',long['input_ids'],params)
        while 'target' not in tokens and e.has_unfinished_requests():
            consume(e.step());steps+=1
            assert steps<10000,'Alternative component exceeded bounded diagnostic'
        quanta=observer.finish();end=time.perf_counter()
        target_work=sum(s['tokens']*s['G'] for q in quanta for s in q['spans']
            if s['id']=='target' and s['phase'] in ('prefill','prefilling_recurrent'))
        row=dict(cell=cell,initial_state=initial,background_active=len(initial),
            target_input_id=long['id'],background_input_ids=job['background_ids'],
            target_TTFT_s=None if not clocks['target'] else clocks['target'][0]-begin,
            intervention_begin=begin,intervention_end=end,output_ids=tokens,token_times=dict(clocks),
            target_prompt_loop_work=target_work,expected_target_prompt_loop_work=4*len(long['input_ids']),
            service_observations=quanta,full_R4=target_work==4*len(long['input_ids']),
            scope='Forced legal fixed C/G diagnostic. Stop after target first output. Not full-answer accuracy or adaptive E2E.',
            gap_scope='Pre-intervention fingerprint hashing pauses the host. Exclude any gap crossing intervention_begin.',
            physical_GPU=guard())
        write(target/f"{cell['label']}.json",row)
        assert row['full_R4'] and row['target_TTFT_s'] is not None
        summaries.append({k:row[k] for k in ('cell','background_active','target_TTFT_s','full_R4')})
        observer.close()
        for rid in list(e.scheduler.requests):e.abort_request(rid)
        e.close();del e,fairness,observer,initial,quanta,row,consume
        gc.collect();torch.cuda.empty_cache()
    return dict(alternative_components=summaries,
        scope='Compare initial KV/hidden hashes before a causal alternative claim. No forced adaptive-policy win.')
