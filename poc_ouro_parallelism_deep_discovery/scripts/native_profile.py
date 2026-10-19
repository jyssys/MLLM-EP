"""Diagnostic native trace with paired clean controls. No headline profiler speedup."""
import argparse
import statistics
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
from poc_ouro_parallelism_deep_discovery.contract import *
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
configure_native_imports()

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--tp',type=int,choices=[1,2,4],required=True);args=p.parse_args()
    mask=validate_mask();assert len(mask)==args.tp
    from vllm import LLM,SamplingParams
    baseline=next((ROOT/'outputs').glob(f'native_base{args.size.replace(".","").replace("B","")}_R4_P{args.tp}_v*/{args.size}_R4_TP{args.tp}/status.json'),None)
    if args.size=='1.4B' and args.tp==2:baseline=ROOT/'outputs/native_base14_R4_P2_v4/1.4B_R4_TP2/status.json'
    assert baseline and read(baseline)['success']
    config=read(baseline.parent/'engine_config.json');workload=read(baseline.parent/'workload.json')['inputs'][0]
    config['worker_extension_cls']='poc_ouro_parallelism_deep_discovery.worker_probe.CampaignProbe'
    out=Path(os.environ['LOOPED_KV_OUTPUT']);write(out/'engine_config.json',config)
    llm=LLM(**config);write(out/'ownership.json',llm.collective_rpc('inspect_ownership',timeout=120))
    teacher=workload['teacher_ids'][:8]
    def generate():
        llm.collective_rpc('teacher',timeout=120,args=(teacher,))
        before=time.perf_counter()
        value=llm.generate([{'prompt_token_ids':workload['prompt_ids']['1024']}],
            SamplingParams(temperature=0,max_tokens=8,ignore_eos=True),use_tqdm=False)[0]
        elapsed=(time.perf_counter()-before)*1000
        assert list(value.outputs[0].token_ids)==teacher
        return elapsed
    for _ in range(3):generate()
    before=[generate() for _ in range(3)]
    llm.collective_rpc('runtime_counts_start',timeout=120)
    llm.collective_rpc('start_kernel_profile',timeout=120)
    traced=generate()
    traces=llm.collective_rpc('finish_kernel_profile',timeout=120)
    counts=llm.collective_rpc('runtime_counts_finish',timeout=120)
    after=[generate() for _ in range(3)]
    write(out/'profile_observation.json',dict(size=args.size,TP=args.tp,R=4,physical_gpus=mask,
        clean_before_ms=before,clean_after_ms=after,profile_ms=traced,
        observer_fraction=traced/statistics.median(before+after)-1,
        traces=traces,counts=counts,output_ids=teacher,output_parity=True,
        evidence_class='GPU_MICROBENCH',
        label='Full native model diagnostic trace. Controlled eight-token work, not natural quality or candidate performance.',
        attribution='GPU service and host events only. Per-rank profiler clocks are not globally aligned. Critical-path attribution remains PARTIAL.',
        clean_contract='Strong optimized graph/custom-All-Reduce configuration reused from the fresh native baseline.'))
    write(out/'status.json',dict(success=True,quality='NOT_RUN',profile_output_parity=True))

if __name__=='__main__':main()
