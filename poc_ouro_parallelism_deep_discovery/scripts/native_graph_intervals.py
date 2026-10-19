"""Observe every graph replay in a request with external CUDA events."""
import argparse
import dataclasses
import statistics
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
from poc_ouro_parallelism_deep_discovery.contract import *
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
configure_native_imports()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',choices=MODELS,required=True)
    parser.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    args=parser.parse_args();mask=validate_mask();assert len(mask)==args.tp
    from vllm import LLM,SamplingParams
    short=args.size.replace('.','').replace('B','')
    base=ROOT/f'outputs/control_async_base{short}_P{args.tp}_v1/{args.size}_R4_TP{args.tp}'
    assert read(base/'status.json')['success']
    config=read(base/'engine_config.json');assert config['async_scheduling']
    config['worker_extension_cls']='poc_ouro_parallelism_deep_discovery.worker_probe.CampaignProbe'
    out=Path(os.environ['LOOPED_KV_OUTPUT']);write(out/'engine_config.json',config)
    workload=read(base/'workload.json')['inputs'][0];teacher=workload['teacher_ids']
    assert len(teacher)==128
    llm=LLM(**config)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership}=={UUIDS[GPUS.index(g)] for g in mask}
    write(out/'ownership.json',ownership)
    def generate(N):
        llm.collective_rpc('teacher',timeout=120,args=(teacher,))
        begin=time.perf_counter()
        value=llm.generate([{'prompt_token_ids':workload['prompt_ids'][str(N)]}],
            SamplingParams(temperature=0,max_tokens=128,ignore_eos=True),use_tqdm=False)[0]
        wall=(time.perf_counter()-begin)*1000
        assert list(value.outputs[0].token_ids)==teacher
        return dict(wall_ms=wall,metrics=dataclasses.asdict(value.metrics),output_ids=list(value.outputs[0].token_ids))
    records=[]
    for context in [1024,4096,8192]:
        for _ in range(3):generate(context)
        before=[generate(context) for _ in range(3)]
        observed=[]
        for repeat in range(3):
            llm.collective_rpc('external_intervals_start',timeout=120)
            value=generate(context)
            ranks=llm.collective_rpc('external_intervals_finish',timeout=120)
            write(out/f'context{context}_repeat{repeat}_intervals.json',dict(context=context,repeat=repeat,
                request=value,ranks=ranks,physical_gpus=mask,evidence_class='GPU_MICROBENCH'))
            observed.append(value)
        after=[generate(context) for _ in range(3)]
        x=statistics.median(r['wall_ms'] for r in before)
        y=statistics.median(r['wall_ms'] for r in after)
        z=statistics.median(r['wall_ms'] for r in observed)
        records.append(dict(context=context,clean_before=before,clean_after=after,observed=observed,
            observer_vs_before_fraction=z/x-1,observer_vs_after_fraction=z/y-1,
            before_after_fraction=y/x-1,output_parity=True))
        write(out/'observation.json',dict(size=args.size,TP=args.tp,records=records,physical_gpus=mask,
            evidence_class='GPU_MICROBENCH',baseline_option='Existing native async scheduling enabled.',
            timing='Every actual replay gets its own event pair. No final-replay multiplication.',
            attribution='Graph internals remain opaque. Nested outer call intervals must not be summed.',
            headroom='Unattributed gaps do not establish removable or recurrence-specific time.',
            quality='Controlled teacher tokens only. No task accuracy or candidate E2E result.'))
        print('GRAPH_INTERVALS',args.size,args.tp,context,z/x-1,z/y-1,flush=True)
    write(out/'status.json',dict(success=True,contexts=3,output_parity=True,quality='NOT_RUN',
        observer_validity='Evaluate paired clean controls before any latency attribution.'))

if __name__=='__main__':main()
