"""Reuse the qualified valid-state barrier for isolated RLT service cohorts."""
import gc
import sys
from .common import *


def main():
    from .local_metrics import prepare
    prepare(config())
    from .runtime import install
    install();cfg=config();folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    import torch
    from vllm_rlt.models import AutoModelForCausalLM
    from vllm_rlt.engine.stage_engine import StageEngine
    from ..looped_pd_abc.worker import configuration
    from ..looped_pd_abc.nvml_watch import GPUWatch
    from ..looped_pd_bottleneck_crossover import capacity_worker as inherited,telemetry
    inherited.ROOT=ROOT;inherited.config=config
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start();engine=None
    try:
        torch.set_num_threads(1);torch.cuda.set_device(0)
        model=AutoModelForCausalLM.from_pretrained(MODEL_PATHS[cfg['model']],device='cuda:0',dtype=torch.bfloat16)
        conf=configuration(cfg['model'],cfg['capacity'],cfg['chunk'],cfg['graphs'],cfg['asynchronous'])
        barrier=inherited.LocalBarrier();engine=StageEngine(model,connector=barrier,**conf)
        cache=engine.cache_manager
        write(folder/'geometry.json',dict(KV_blocks=cache.num_blocks,KV_bytes=cache.key_cache.numel()*cache.key_cache.element_size()*2,
            KV_block_bytes=cache.bytes_per_block,block_size=cache.block_size,layers=cache.num_layers,storage_depths=cache.storage_depths,
            KV_heads=cache.num_kv_heads,head_dim=cache.head_dim,unique_parameters=sum(p.numel() for p in model.parameters()),
            weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),numerical_variant='down_norm' if cfg['model']=='ouro' else 'down_norm_adapter'))
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
        for j,job in enumerate(cfg['jobs']):
            batch=job['batch'];cap=job.get('cap',cfg['cap']);tokens=[r['input_ids'][cfg['model']] for r in data[:batch]]
            assert len(tokens)==batch and all(len(t)+cap<=cfg['context_limit'] for t in tokens)
            needed=sum(cache.required_blocks(len(t)+cap-1) for t in tokens)
            if needed>cache.num_blocks or batch>cfg['capacity']:
                append(folder/'cohorts.jsonl',dict(status='BLOCKED_KV_OR_SLOTS',job=job,required_blocks=needed,available_blocks=cache.num_blocks));continue
            recorder=telemetry.Recorder(enabled=False);telemetry.ACTIVE=None
            inherited.cohort(engine,tokens,min(2,cap),cfg['seed']-10000,f'j{j}_warm',barrier,recorder)
            reference=inherited.cohort(engine,tokens,cap,cfg['seed'],f'j{j}_reference',barrier,recorder,use_barrier=False)
            append(folder/'references.jsonl',dict(job=job,**reference))
            for repeat in range(job.get('repeats',3)):
                value=inherited.cohort(engine,tokens,cap,cfg['seed'],f'j{j}_r{repeat}',barrier,recorder)
                def index(rows):return {k.rsplit('_q',1)[1]:v for k,v in rows.items()}
                assert index(value['outputs'])==index(reference['outputs']),'Native/barrier output mismatch'
                value.update(model=cfg['model'],job=job,repeat=repeat,chunk=cfg['chunk'],KV_blocks=cfg['kv_blocks'],
                    status='COMPLETE',native_output_parity=True,capacity_scope='Finite valid-state cohort, not sustainable Serving capacity.',
                    evidence='MEASURED_UNPROFILED_INDEPENDENT_SERVICE')
                append(folder/'cohorts.jsonl',value);print('COHORT_COMPLETE',cfg['name'],j,repeat,flush=True)
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if engine is not None:engine.close()
        monitor.stop()


if __name__=='__main__':main()
