"""Order only newly allocated KV blocks before assigning logical positions."""
import atexit,time
from operator import attrgetter
from .common import ROOT,write,digest,json


def canonicalize(blocks):
    blocks.sort(key=attrgetter('block_id'))
    return blocks


def install(cfg,role):
    if not cfg.get('canonical_new_KV_block_order'):return
    contract=json.loads((ROOT/'state/canonical_block_runtime_contract.json').read_text())
    for path,expected in contract['hashes'].items():
        assert digest(path)==expected,'The canonical allocation contract changed: '+path
    from vllm.v1.core.block_pool import BlockPool
    original=BlockPool.get_new_blocks
    statistics=dict(calls=0,blocks=0,sort_host_total_ms=0.,sort_host_max_ms=0.)
    examples=[]
    def allocate(self,num_blocks):
        assert not self.enable_caching,'Qualify cached/shared-block allocation separately'
        blocks=original(self,num_blocks)
        assert all(not b.is_null and b.ref_cnt==1 for b in blocks)
        before=[b.block_id for b in blocks] if cfg.get('correctness_only') else None
        begin=time.monotonic();canonicalize(blocks);elapsed=1000*(time.monotonic()-begin)
        if before is not None:
            assert sorted(before)==[b.block_id for b in blocks]
            assert len(set(before))==len(blocks)
            if len(examples)<64:examples.append(dict(before=before,after=[b.block_id for b in blocks]))
        statistics['calls']+=1;statistics['blocks']+=len(blocks)
        statistics['sort_host_total_ms']+=elapsed
        statistics['sort_host_max_ms']=max(statistics['sort_host_max_ms'],elapsed)
        return blocks
    BlockPool.get_new_blocks=allocate
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/f'canonical_new_blocks_{role}.json',
        dict(statistics=statistics,examples=examples,evidence='STRUCTURAL_ACCOUNTING',
            scope='The original allocator selects, owns and references the same block set. Sort only its newly allocated return list before logical positions are assigned. Existing request blocks, free-queue links, counts, ACK and retirement are unchanged. Prefix caching is disabled. Counters include warm-up. This is a generic PD layout change, not a Dynamic scheduling gain.')))
