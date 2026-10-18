"""Physical native CDB batch/context cost samples. Synthetic prompts are diagnostics only."""
import os
import time
import torch
from poc_decode_first_depth_aware_dp.runtime import load_engine,stats_dict
from poc_decode_first_depth_aware_dp_v3.contract import *

@torch.inference_mode()
def main():
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    model,tok,engine=load_engine(graphs=True,batch=16,blocks=4096,max_model_len=10240)
    base=tok.encode('Explain how a computer executes a program. ',add_special_tokens=False)
    rows=[]
    for context in [1024,4096,8192]:
        ids=(base*(context//len(base)+1))[:context]
        for batch in [1,2,4,8,16,32]:
            if batch>16 or batch*(context+32)>4096*16:
                rows.append(dict(context=context,batch=batch,status='SKIPPED_CAPACITY',
                    reason='Resident cap 16 or explicit 65536-token physical KV budget.'))
                continue
            for repeat in range(4):
                torch.cuda.synchronize();start=time.perf_counter()
                values=engine.generate_batch([ids]*batch,max_new_tokens=32,
                    eos_token_id=tok.eos_token_id,warmup=repeat==0)
                torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                stats=stats_dict(engine.last_stats)
                row=dict(context=context,batch=batch,repeat=repeat,warmup=repeat==0,
                    status='MEASURED_SYNTHETIC_PROMPT',wall_ms=wall,
                    mean_decode_span_ms=sum((v.lifespan[1]-v.first_token_time)*1000 for v in values)/batch,
                    mean_TTFT_ms=sum((v.first_token_time-v.created_time)*1000 for v in values)/batch,
                    executed_loops=stats['recurrent_steps'],recurrent_batches=stats['recurrent_batches'],
                    mean_active_batch=stats['recurrent_steps']/stats['recurrent_batches'],
                    refill_events=stats['scheduler_refills'],
                    memory_allocated=torch.cuda.memory_allocated(),peak_memory=torch.cuda.max_memory_allocated())
                rows.append(row)
                write(out/'cost_surface.json',rows)
                print('COST_SURFACE',context,batch,repeat,round(wall,2),flush=True)
    write(out/'cost_surface.json',rows)
    write(out/'status.json',dict(success=True,category='MEASURED_PHYSICAL_CDB_SYNTHETIC_CONTEXT_DIAGNOSTIC'))

if __name__=='__main__':main()
