"""Reuse existing native Mixed for a qualified colocated control, not a new method."""
import collections
import sys
from dataclasses import replace
from .common import *


def worker_mapping(infos,gpus):
    mapping=[]
    for index,info in enumerate(infos):
        actual=info['cuda_uuid'];gpu=gpus[index]
        assert actual.removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
        mapping.append(dict(replica=index,physical_GPU=gpu,cuda_uuid=actual,fingerprint=info.get('fingerprint','NOT_EXPOSED_WITHOUT_NIXL_CONNECTOR')))
    return mapping


def install():
    from vllm_rlt.engine.stage_engine import StageEngine
    if getattr(StageEngine,'_atomic_native_mixed_installed',False):return
    StageEngine._atomic_native_mixed_installed=True
    cfg=config();hashes=json.loads(os.environ.get('ATOMIC_SOURCE_HASHES','{}'))
    for path,expected in hashes.items():assert digest(path)==expected,'Native Mixed source changed at startup'
    os.environ.update(ABC_RUN_DIR=str(ROOT/'raw'/cfg['name']),B_NUMERICAL_RUN=cfg['name'],
        B_NUMERICAL_VARIANT='down_norm' if cfg['model']=='ouro' else 'down_norm_adapter')
    from ..looped_pd_abc import common as base_common
    base_common.ROOT=ROOT;base_common.setup=setup
    from ..looped_pd_b_numerical_diagnosis import common as numerical
    numerical.ROOT=ROOT
    from ..looped_pd_b_numerical_diagnosis.patches import install as qualified
    qualified()
    from ..looped_pd_abc import worker as old
    configure=old.configuration
    def configuration(model,width,chunk,graphs=False,asynchronous=False):
        value=configure(model,width,chunk,graphs,asynchronous)
        value['cache_config']=replace(value['cache_config'],num_blocks=cfg['kv_blocks'],enable_prefix_caching=False)
        value['scheduler_config']=replace(value['scheduler_config'],max_num_batched_tokens=cfg['token_budget'],
            prefill_chunk_size=cfg['chunk'],mode='mixed',wavefront_chunked_prefill=cfg.get('wavefront',True))
        value['execution_config']=replace(value['execution_config'],cuda_graph_max_batch_size=cfg.get('graph_rows',cfg['capacity']))
        return value
    old.configuration=configuration
    from ..looped_pd_abc.replicas import ReplicaPool
    initialize=ReplicaPool.__init__
    def replicas(self,*args,**kwargs):
        initialize(self,*args,**kwargs)
        try:write(ROOT/'raw'/cfg['name']/'native_mixed_worker_map.json',worker_mapping([c.info for c in self.clients],cfg['gpus']))
        except BaseException:self.close();raise
    ReplicaPool.__init__=replicas
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.request import RequestPhase
    submit=ModelRunner.submit;close=StageEngine.close;counts=collections.Counter();examples=[]
    def observed(self,scheduled_batch=None,*,batch):
        phases=[scheduled_batch.phase_for(item) for item in scheduled_batch.requests] if scheduled_batch is not None else []
        mixed=RequestPhase.RECURRENT in phases and RequestPhase.PREFILLING_RECURRENT in phases
        counts[(batch.operation.value,batch.num_rows,batch.loop_count,len(batch.spans),mixed)]+=1
        if mixed and len(examples)<8:
            examples.append(dict(time=time.monotonic(),logical_rows=batch.num_rows,loop_count=batch.loop_count,
                spans=[dict(seed=s.request.sampling_params.seed,start=s.token_start,count=s.token_count,depth=s.depth,phase=p.value)
                    for s,p in zip(batch.spans,phases)]))
        return submit(self,scheduled_batch,batch=batch)
    def closed(self):
        self.model_runner.synchronize()
        write(ROOT/'raw'/cfg['name']/f'native_mixed_execution_{os.getpid()}.json',dict(role=self.stage_config.role,hashes=hashes,
            histogram=[dict(operation=k[0],logical_rows=k[1],loop_count=k[2],requests=k[3],mixed=k[4],submissions=v) for k,v in counts.items()],
            examples=examples,scope='Existing native Mixed scheduler and shared-core batching. Includes warm-up. Not a new RCC or Dynamic-A mechanism.'))
        return close(self)
    ModelRunner.submit=observed;StageEngine.close=closed


def main():
    install();cfg=config()
    from ..looped_pd_abc import worker as old
    from .rlt_worker import serve
    old.ROOT=ROOT;old.setup=setup;old.serve=serve
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    old.prompts=lambda model,count,length:[r['input_ids'][model] for r in data[:count]]
    write(ROOT/'raw'/cfg['name']/'native_mixed_contract.json',cfg)
    flags=['worker','--model',cfg['model'],'--mode','dp','--gpus',','.join(map(str,cfg['gpus'])),
        '--name',cfg['name'],'--requests',str(cfg['count']),'--length','0','--cap',str(cfg['cap']),
        '--chunk',str(cfg['chunk']),'--concurrency',str(cfg['capacity']),'--timeout',str(cfg['timeout']),
        '--scheduler-mode','mixed','--warmup']
    if cfg.get('wavefront',True):flags.append('--wavefront')
    if cfg['graphs']:flags.append('--graphs')
    if cfg['asynchronous']:flags.append('--asynchronous')
    sys.argv=flags;old.main()


if __name__=='__main__':main()
