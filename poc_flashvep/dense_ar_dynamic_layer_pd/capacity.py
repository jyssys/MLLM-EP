"""Reuse Native local calibration without representing it as online PD."""
import argparse,json,os,time
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ['ARC_CONFIG']=a.config
    if cfg.get('kv_layout'):os.environ['VLLM_KV_CACHE_LAYOUT']=cfg['kv_layout']
    os.environ['VLLM_BATCH_INVARIANT']='1'
    from ..looped_pd_atomic_prefill import ar_worker,ar_capacity
    ar_worker.MODEL_PATHS=dict(qwen8=MODEL);ar_worker.ROOT=ROOT
    if cfg.get('fixed_KV_bytes') or cfg.get('ar_compile_mode') is not None or cfg.get('ar_custom_ops'):
        original=ar_worker.llm_options
        def options(config,role):
            value=original(config,role)
            if config.get('fixed_KV_bytes'):value['kv_cache_memory_bytes']=config['fixed_KV_bytes']
            if config.get('ar_compile_mode') is not None:value['compilation_config']['mode']=config['ar_compile_mode']
            if config.get('ar_custom_ops'):value['compilation_config']['custom_ops']=config['ar_custom_ops']
            if config.get('decode_only_graph_roles') and role not in config['decode_only_graph_roles']:
                value['compilation_config']['cudagraph_mode']='NONE'
            return value
        ar_worker.llm_options=options
    ar_capacity.ROOT=ROOT
    if cfg.get('runtime_observation') or cfg.get('layer_mode') or cfg.get('prime_detokenizer'):
        from .extension import install
        install(cfg,'capacity')
    try:
        ar_capacity.main()
    finally:
        folder=ROOT/'raw'/cfg['name'];state=folder/'status.json'
        passed=state.exists() and json.loads(state.read_text()).get('status')=='COMPLETE'
        write(folder/'worker_exits.json',dict(time=time.time(),passed=passed,
            scope='Local calibration returned after Native engine shutdown. Not two-worker transport qualification.'))


if __name__=='__main__':main()

