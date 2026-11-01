"""Run a normal qualified local engine as the finite output/state reference."""
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
    from ..looped_pd_bottleneck_crossover.capacity_worker import LocalBarrier,cohort
    from ..looped_pd_bottleneck_crossover import capacity_worker as previous,telemetry
    previous.ROOT=ROOT;previous.config=config
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start();engine=None
    try:
        torch.set_num_threads(1);torch.cuda.set_device(0)
        model=AutoModelForCausalLM.from_pretrained(MODEL_PATHS[cfg['model']],device='cuda:0',dtype=torch.bfloat16)
        barrier=LocalBarrier();engine=StageEngine(model,connector=barrier,**configuration(cfg['model'],cfg['capacity'],cfg['chunk'],cfg['graphs'],cfg['asynchronous']))
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
        tokens=[r['input_ids'][cfg['model']] for r in data];recorder=telemetry.Recorder(enabled=False)
        cohort(engine,tokens[:2],2,cfg['seed']-10000,'warm',barrier,recorder,use_barrier=False)
        result=cohort(engine,tokens,cfg['cap'],cfg['seed'],'reference',barrier,recorder,use_barrier=False)
        append(folder/'references.jsonl',dict(job=dict(batch=cfg['count'],cap=cfg['cap'],dataset=cfg['dataset']),**result))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if engine is not None:engine.close()
        monitor.stop()


if __name__=='__main__':main()
