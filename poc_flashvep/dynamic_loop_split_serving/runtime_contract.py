"""Record native runtime settings outside the measured serving interval."""
import os
from .common import ROOT, write


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    initialize, close = StageEngine.__init__, StageEngine.close

    def snapshot(engine, phase):
        cache = engine.cache_manager
        scheduler = engine.scheduler.config
        return dict(phase=phase, role=engine.stage_config.role,
            model_revision=cfg['model_revision'], fixed_depth=cfg['depth'], dtype=str(cache.dtype),
            prefix_caching_enabled=cache.enable_prefix_caching,
            reused_prefix_tokens=cache.prefix_hits, allocation_queries=cache.prefix_queries,
            scheduler_mode=scheduler.mode, wavefront=scheduler.wavefront_chunked_prefill,
            chunk_size=scheduler.prefill_chunk_size, token_budget=scheduler.max_num_batched_tokens,
            max_sequences=scheduler.max_num_seqs, KV_blocks=cache.num_blocks,
            used_KV_blocks=cache.num_used_blocks,
            scope='Native settings and counters. Allocation queries are not cache hits. No prefix reuse is allowed in this campaign.')

    def initialized(engine, *args, **kwargs):
        value = initialize(engine, *args, **kwargs)
        assert not engine.cache_manager.enable_prefix_caching, 'Main serving requires cold/no-prefix-reuse execution'
        write(ROOT / 'raw' / cfg['name'] / f'native_contract_initial_{os.getpid()}.json', snapshot(engine, 'INITIAL'))
        return value

    def closed(engine):
        assert engine.cache_manager.prefix_hits == 0, 'Unexpected prompt reuse'
        write(ROOT / 'raw' / cfg['name'] / f'native_contract_final_{os.getpid()}.json', snapshot(engine, 'FINAL'))
        return close(engine)

    StageEngine.__init__, StageEngine.close = initialized, closed
