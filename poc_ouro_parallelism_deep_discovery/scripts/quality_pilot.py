"""Mode-matched natural greedy Base-model pilot. No candidate mechanism is enabled."""
import argparse
import collections
import datetime
import gc
import signal
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports,load_engine
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--mode',choices=['fixed','adaptive'],required=True)
    args=p.parse_args();out=Path(os.environ['LOOPED_KV_OUTPUT'])/f'{args.size}_{args.mode}'
    out.mkdir(parents=True,exist_ok=False)
    protocol=read(ROOT/'artifacts/probe_protocol.json')
    rows=[r for r in protocol['rows'] if r['campaign_partition']=='calibration']
    assert len(rows)==16 and all(sum(r['dataset']==d for r in rows)==8 for d in ['gsm8k','math500'])
    model,tok,engine=load_engine(args.size,args.mode,graphs=True,batch=1,blocks=640,max_model_len=10240)
    engine.generate_batch([tok.encode('Q: What is 1 + 1?\nA:')],max_new_tokens=4,
        eos_token_id=tok.eos_token_id,warmup=True)
    depths=[];original=engine._record_exit
    def recorded(item,depth):
        depths.append({'position':item.token_position,'depth':depth+1})
        return original(item,depth)
    engine._record_exit=recorded
    def watchdog(signum,frame):raise TimeoutError('A natural pilot request exceeded 1200 seconds.')
    signal.signal(signal.SIGALRM,watchdog)
    from poc_decode_first_depth_aware_dp.runtime import stats_dict
    for row in rows:
        ids=tok.encode(row['prompt'],add_special_tokens=False)
        assert len(ids)+protocol['quality_cap']<=10240
        stops=[tok.encode(s,add_special_tokens=False) for s in row.get('stop_strings',[])]
        depths.clear();torch.cuda.synchronize();before=time.perf_counter()
        signal.alarm(1200)
        try:
            value=engine.generate_batch([ids],max_new_tokens=protocol['quality_cap'],
                eos_token_id=tok.eos_token_id,stop_sequences=stops,warmup=False)[0]
        finally:signal.alarm(0)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-before)*1000
        tokens=value.generated_tokens
        assert len(depths)==len(tokens)-1
        if args.mode=='fixed':assert all(r['depth']==4 for r in depths)
        else:assert all(r['depth'] in [2,3,4] for r in depths)
        reason='eos' if tokens and tokens[-1]==tok.eos_token_id else 'length' if len(tokens)==protocol['quality_cap'] else 'stop'
        result=dict(id=row['id'],dataset=row['dataset'],size=args.size,mode=args.mode,
            model=MODELS[args.size]['id'],revision=MODELS[args.size]['revision'],physical_gpu=validate_mask()[0],
            prompt_ids=ids,output_ids=tokens,text=tok.decode(tokens,skip_special_tokens=False),
            generated_tokens=len(tokens),finish_reason=reason,depth_events=list(depths),
            stats=stats_dict(engine.last_stats),wall_ms=elapsed,
            first_output_source='Full R4 prefill. Decode depths exclude this first token.',
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            EOS=tok.eos_token_id,cap=protocol['quality_cap'],temperature=0,
            label='QUALITY_ONLY baseline pilot. Not candidate performance or distributed architecture speedup.')
        write(out/(row['id'].replace('/','_')+'.json'),result)
        print('PILOT',args.size,args.mode,row['id'],len(tokens),reason,collections.Counter(d['depth'] for d in depths),flush=True)
    write(out/'status.json',dict(success=True,N=16,per_domain=8,mode=args.mode,size=args.size,
        source_protocol_sha256=sha(ROOT/'artifacts/probe_protocol.json'),quality_scope='Calibration pilot, not confirmation.'))

if __name__=='__main__':main()
