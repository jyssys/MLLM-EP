"""Bounded native KV capture for an exact linear-certificate falsifier."""
import argparse
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
from poc_ouro_parallelism_deep_discovery.contract import *
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
configure_native_imports()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',choices=MODELS,required=True)
    args=parser.parse_args();mask=validate_mask();assert len(mask)==1
    from vllm import LLM,SamplingParams
    short=args.size.replace('.','').replace('B','')
    base=ROOT/f'outputs/control_async_base{short}_P1_v1/{args.size}_R4_TP1'
    assert read(base/'status.json')['success']
    config=read(base/'engine_config.json')
    config['worker_extension_cls']='poc_ouro_parallelism_deep_discovery.worker_probe.CampaignProbe'
    output=Path(os.environ['LOOPED_KV_OUTPUT']);write(output/'engine_config.json',config)
    workload=read(base/'workload.json')['inputs'][0];teacher=workload['teacher_ids'][:4]
    llm=LLM(**config)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(ownership)==1 and ownership[0]['cache_slots']==MODELS[args.size]['layers']*4
    assert ownership[0]['GPU_uuid'].removeprefix('GPU-')==UUIDS[GPUS.index(mask[0])]
    write(output/'ownership.json',ownership)
    llm.collective_rpc('teacher',timeout=120,args=(teacher,))
    result=llm.generate([{'prompt_token_ids':workload['prompt_ids']['1024']}],
        SamplingParams(temperature=0,max_tokens=4,ignore_eos=True),use_tqdm=False)[0]
    assert list(result.outputs[0].token_ids)==teacher
    captures=llm.collective_rpc('export_key_rank_fixture',timeout=120,args=(257,))
    write(output/'status.json',dict(success=True,size=args.size,model=MODELS[args.size]['id'],
        revision=MODELS[args.size]['revision'],physical_gpu=mask[0],captures=captures,
        source_workload=str(base/'workload.json'),source_workload_sha256=sha(base/'workload.json'),
        output_ids=teacher,evidence_class='QUALITY_ONLY',
        scope='Actual native BF16 KV values. Controlled request, not benchmark accuracy. Capture latency is not a headline timing.'))

if __name__=='__main__':main()
