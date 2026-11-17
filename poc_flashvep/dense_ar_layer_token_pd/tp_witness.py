"""Trace the same prompt and common greedy prefix in TP1 and TP2."""
import json
import os
import sys
import time
from .common import MODEL, ROOT, write


def main():
    cfg = json.load(open(sys.argv[1]))
    folder = ROOT / 'raw' / cfg['name']
    write(folder / 'status.json', dict(status='RUNNING', time=time.time()))
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    import torch
    from vllm import LLM, SamplingParams
    from ..dense_ar_slo_layer_pd.watch import GPUWatch
    monitor = GPUWatch(folder, cfg['gpus'], cfg['name'])
    monitor.start()
    llm = None
    normal = False
    try:
        torch.set_num_threads(1)
        llm = LLM(model=str(MODEL), dtype='bfloat16', tensor_parallel_size=cfg['tp'],
            distributed_executor_backend='mp' if cfg['tp'] > 1 else 'uni',
            worker_cls='poc_flashvep.dense_ar_layer_token_pd.tensor_worker.TensorWitnessWorker',
            max_model_len=cfg['length'] + cfg['cap'], max_num_seqs=1,
            max_num_batched_tokens=8192, enable_chunked_prefill=True,
            enable_prefix_caching=False, async_scheduling=False,
            kv_cache_memory_bytes=cfg['fixed_KV_bytes'], seed=cfg['seed'],
            compilation_config=dict(mode=0, custom_ops=['all'], cudagraph_mode='NONE'))
        write(folder / 'worker_reports.json', llm.collective_rpc('duetserve_report', timeout=60))
        write(folder / 'capture_install.json', llm.collective_rpc('witness_enable', timeout=120))
        data = json.loads((ROOT / 'data' / cfg['dataset']).read_text())
        prompt = data[cfg['sample_id']]['input_ids']['qwen8']
        outputs = llm.generate([{'prompt_token_ids': prompt}],
            SamplingParams(temperature=0, max_tokens=cfg['cap'], ignore_eos=True), use_tqdm=False)
        write(folder / 'greedy_output.json', dict(sample_id=cfg['sample_id'], input_ids=prompt,
            output_ids=list(outputs[0].outputs[0].token_ids), scope='CORRECTNESS_ONLY'))
        write(folder / 'capture_export.json', llm.collective_rpc('witness_export', timeout=120))
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        assert not scheduler.requests
        llm.llm_engine.engine_core.shutdown()
        normal = True
        write(folder / 'status.json', dict(status='COMPLETE', time=time.time()))
    finally:
        monitor.stop()
        write(folder / 'worker_exits.json', dict(passed=normal))


if __name__ == '__main__':
    main()
