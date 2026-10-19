"""Low-overhead whole-graph diagnostic. Not a replacement for clean E2E."""
import argparse
import dataclasses
import statistics
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
from poc_ouro_parallelism_deep_discovery.contract import *
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
configure_native_imports()

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--tp',type=int,choices=[2,4],required=True);args=p.parse_args()
    mask=validate_mask();assert len(mask)==args.tp
    from vllm import LLM,SamplingParams
    import torch
    short=args.size.replace('.','').replace('B','')
    suffix='v4' if short=='14' and args.tp==2 else 'v1'
    base=ROOT/f'outputs/native_base{short}_R4_P{args.tp}_{suffix}/{args.size}_R4_TP{args.tp}'
    assert read(base/'status.json')['success']
    config=read(base/'engine_config.json')
    config['worker_extension_cls']='poc_ouro_parallelism_deep_discovery.worker_probe.CampaignProbe'
    workload=read(base/'workload.json')['inputs'][0]
    out=Path(os.environ['LOOPED_KV_OUTPUT']);write(out/'engine_config.json',config)
    llm=LLM(**config);write(out/'ownership.json',llm.collective_rpc('inspect_ownership',timeout=120))
    teacher=workload['teacher_ids'][:16]
    request_metrics=[]
    def generate(label):
        llm.collective_rpc('teacher',timeout=120,args=(teacher,))
        torch.cuda.nvtx.range_push(label)
        before=time.perf_counter()
        try:
            value=llm.generate([{'prompt_token_ids':workload['prompt_ids']['1024']}],
                SamplingParams(temperature=0,max_tokens=16,ignore_eos=True),use_tqdm=False)[0]
        finally:torch.cuda.nvtx.range_pop()
        elapsed=(time.perf_counter()-before)*1000
        assert list(value.outputs[0].token_ids)==teacher
        request_metrics.append(dict(label=label,wall_ms=elapsed,
            metrics=None if value.metrics is None else dataclasses.asdict(value.metrics)))
        return elapsed
    for _ in range(3):generate('warmup_outside_capture')
    before=[generate('control_before_capture') for _ in range(3)]
    starts=llm.collective_rpc('profiler_api_start',timeout=120)
    observed=generate('OBSERVED_REQUEST')
    stops=llm.collective_rpc('profiler_api_stop',timeout=120)
    after=[generate('control_after_capture') for _ in range(3)]
    write(out/'profile_observation.json',dict(size=args.size,TP=args.tp,R=4,physical_gpus=mask,
        clean_before_ms=before,profile_ms=observed,clean_after_ms=after,
        observer_vs_before_fraction=observed/statistics.median(before)-1,
        observer_vs_after_fraction=observed/statistics.median(after)-1,
        api_start=starts,api_stop=stops,output_ids=teacher,output_parity=True,
        request_metrics=request_metrics,
        label='Whole-graph Nsight Systems diagnostic with node tracing disabled.',
        evidence_class='GPU_MICROBENCH',
        limitation='Graph-internal operator time remains opaque. Parent wall and CUDA streams are different intervals.',
        scope='Sixteen controlled output tokens at1K. Not candidate performance or natural quality.'))
    write(out/'status.json',dict(success=True,headline=False,scope='Diagnostic trace only'))

if __name__=='__main__':main()
