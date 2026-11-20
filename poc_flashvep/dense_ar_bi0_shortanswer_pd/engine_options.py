"""Keep engine settings explicit across standard Native and Agg workers."""
from .common import MODEL


def options(cfg,role='single'):
    value=dict(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=cfg.get('context_limit',40960),
        max_num_batched_tokens=cfg.get('token_budget',8192),max_num_seqs=cfg.get('max_num_seqs',1024),
        kv_cache_memory_bytes=42*1024**3,enable_prefix_caching=False,enable_chunked_prefill=True,
        async_scheduling=cfg.get('async_scheduling',True),seed=cfg.get('seed',2026111901),
        disable_log_stats=True,attention_config=dict(backend='FLASH_ATTN'))
    if cfg.get('execution_family')=='MATCHED_NO_COMPILE':
        value['compilation_config']=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',
            cudagraph_capture_sizes=[1,2,4,8,16,32])
    if 'KV_block_tokens' in cfg:value['block_size']=int(cfg['KV_block_tokens'])
    if role in ('prefill','decode'):
        value['kv_transfer_config']=dict(kv_connector='NixlConnector',
            kv_role='kv_producer' if role=='prefill' else 'kv_consumer',
            kv_load_failure_policy='fail')
        if 'nixl_num_threads' in cfg:
            value['kv_transfer_config']['kv_connector_extra_config'] = dict(num_threads=cfg['nixl_num_threads'])
    return value
