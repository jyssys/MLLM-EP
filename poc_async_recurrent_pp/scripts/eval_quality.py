"""Free-running single-GPU candidate evaluation. No distributed speedup claim."""
import argparse
import gc
import os
import torch
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.emulator import StageEmulator, generate_batch
from poc_async_recurrent_pp.scripts.baseline import load, token_ids
from poc_async_recurrent_pp.scoring import score


def filename(identifier):
    return identifier.replace('/','_')+'.json'


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['exact','stale','delta','w0'],required=True)
    parser.add_argument('--per-dataset',type=int,default=16)
    parser.add_argument('--cap',type=int,default=2048)
    parser.add_argument('--expand-capped-only',type=Path)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    assert (ROOT/'outputs/baseline_v1/status.json').exists(), 'Exact reference certification is required'
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    torch.manual_seed(20261017)
    model, tokenizer=load()
    data=read(ROOT/'outputs/manifests/data.json')
    coefficients=read(ROOT/'outputs/manifests/predictor.json')
    rows=data['full'] if args.full else sum([[r for r in data['pilot'] if r['dataset']==d][:args.per_dataset]
                                           for d in ['gsm8k','math500']],[])
    if args.expand_capped_only:
        rows=[r for r in rows if read(args.expand_capped_only/filename(r['id']))['truncated']]
    write(output/'eval_contract.json',dict(arm=args.arm,cap=args.cap,IDs=[r['id'] for r in rows],
        R=4,exact_prefill=True,warmup=0 if args.arm=='w0' else 1 if args.arm!='exact' else None,
        GPU_mask=os.environ['CUDA_VISIBLE_DEVICES'],UUID=str(torch.cuda.get_device_properties(0).uuid),
        predictor_sha256=sha(ROOT/'outputs/manifests/predictor.json'),
        data_sha256=sha(ROOT/'outputs/manifests/data.json'),
        dtype='BF16',sampling='Greedy',tokenizer_revision=REVISION,
        source='Official HF forward with boundary-input hooks. Serial functional emulation only.'))
    assert read(ROOT/'outputs/baseline_tasks_v1/status.json')['baseline_plausible']
    records=[]
    for start in range(0,len(rows),8):
        group=rows[start:start+8]
        assert all(not (output/filename(row['id'])).exists() for row in group), 'Preserve previous attempts'
        hook=StageEmulator(model,args.arm,coefficients)
        try:
            prompts=[token_ids(tokenizer,row,data['prompt_suffix']) for row in group]
            results=generate_batch(model,tokenizer,prompts,args.cap,hook)
            if not records:
                write(output/'availability_trace.json',hook.events)
            for row,ids,result in zip(group,prompts,results):
                record=dict(id=row['id'],dataset=row['dataset'],arm=args.arm,prompt_ids=ids,
                    prompt=row['question']+data['prompt_suffix'],cap=args.cap,
                    **result,**score(row,result))
                write(output/filename(row['id']),record)
                records.append(record)
                print(args.arm,row['id'],record['correct_completed'],record['generated_tokens'],record['finish_reason'],flush=True)
        finally:
            hook.close()
            gc.collect()
    write(output/'status.json',dict(success=True,n=len(records),arm=args.arm,
        correct=sum(r['correct_completed'] for r in records),
        parsed=sum(r['parsed'] for r in records),truncated=sum(r['truncated'] for r in records),
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    main()
