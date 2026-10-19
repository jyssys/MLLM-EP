"""Dump actual replayed native CUDA graphs. No performance claim."""
import argparse
import os
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
configure_native_imports()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',choices=MODELS,required=True)
    parser.add_argument('--tp',choices=[1,2,4],type=int,required=True)
    parser.add_argument('--edge-metadata',action='store_true',help='Export read-only CUDA12.9 typed edge metadata after the request.')
    args=parser.parse_args();mask=validate_mask();assert len(mask)==args.tp
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    os.environ['VLLM_CACHE_ROOT']=str(output/'vllm_cache')
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(output/'inductor_cache')
    os.environ['TRITON_CACHE_DIR']=str(output/'triton_cache')
    from vllm import LLM,SamplingParams
    short=args.size.replace('.','').replace('B','')
    source=ROOT/f'outputs/control_async_base{short}_P{args.tp}_v1/{args.size}_R4_TP{args.tp}'
    config=read(source/'engine_config.json');assert config['async_scheduling']
    config['worker_extension_cls']='poc_ouro_parallelism_deep_discovery.graph_debug_probe.GraphDebugProbe'
    if args.edge_metadata:
        config['worker_extension_cls']='poc_ouro_parallelism_deep_discovery.graph_typed_edge_probe.TypedEdgeGraphDebugProbe'
    config['compilation_config']['cache_dir']=str(output/'compile_cache')
    write(output/'engine_config.json',config)
    row=read(source/'workload.json')['inputs'][0];teacher=row['teacher_ids'][:4]
    llm=LLM(**config)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership}=={UUIDS[GPUS.index(g)] for g in mask}
    assert all(r['cache_slots']==4*MODELS[args.size]['layers'] for r in ownership)
    write(output/'ownership.json',ownership)
    def generate():
        llm.collective_rpc('teacher',timeout=120,args=(teacher,))
        result=llm.generate([{'prompt_token_ids':row['prompt_ids']['1024']}],
                            SamplingParams(temperature=0,max_tokens=4,ignore_eos=True),use_tqdm=False)[0]
        assert list(result.outputs[0].token_ids)==teacher
        return list(result.outputs[0].token_ids)
    for _ in range(3):generate()
    llm.collective_rpc('graph_debug_begin',timeout=120)
    llm.collective_rpc('start_parity_capture',timeout=120)
    ids=generate()
    capture=llm.collective_rpc('finish_parity_capture',timeout=120,args=('graph_debug_capture',1024))
    write(output/'parity_capture.json',dict(capture=capture,output_ids=ids,context=1024))
    if args.edge_metadata:
        graphs=llm.collective_rpc('graph_debug_export',timeout=120,args=(MODELS[args.size]['layers'],))
    else:
        graphs=llm.collective_rpc('graph_debug_export',timeout=120)
    for rank in graphs:assert rank['total_observed_replays']==3,rank
    write(output/'topology.json',dict(model=MODELS[args.size]['id'],revision=MODELS[args.size]['revision'],
          R=4,P=args.tp,physical_gpus=mask,graphs=graphs,capture=capture,output_ids=ids,
          request_id=row['id'],context=1024,generated_tokens=4,
          evidence_class='QUALITY_ONLY',
          scope='Native graph topology diagnostic. No model arithmetic replacement. Debug mode and capture observer exclude headline timing.',
          numerical_comparison='Captured only. Compare independently before any parity claim.',
          timing='NOT_MEASURED. DOT edges and kernel counts are not elapsed time.'))
    write(output/'status.json',dict(success=True,native_graph_topology=True,
          candidate_speedup='NOT_RUN',natural_quality='NOT_RUN'))


if __name__=='__main__':main()
