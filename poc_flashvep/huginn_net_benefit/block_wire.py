"""Trade extra invalid padding bytes for fewer exact Native KV copy descriptors."""
import os
from .common import ROOT, write


def owned_frontier_blocks(tables, layout, block_size):
    if block_size <= 0:
        raise ValueError('Block size must be positive')
    all_blocks = [block for table in tables for block in table]
    if len(all_blocks) != len(set(all_blocks)):
        raise ValueError('Depth tables must not alias owned blocks')
    ends, planes = {}, set()
    for depth, layer, end in layout:
        if (depth, layer) in planes:
            raise ValueError('Duplicate semantic KV plane')
        planes.add((depth, layer))
        if not 0 <= depth < len(tables) or end < 0 or end > len(tables[depth])*block_size:
            raise ValueError('The semantic frontier exceeds owned pages')
        ends[depth] = max(end, ends.get(depth, 0))
    return [block for depth in sorted(ends)
        for block in tables[depth][:(ends[depth]+block_size-1)//block_size]]


def install(cfg):
    assert not cfg.get('progressive_kv') and not cfg.get('native_loop_event_transfer'), (
        'Whole-block writes cannot overlap a progressive frontier without a separate dependency proof')
    from ..native_loop_2d_oracle import native_path
    connector = native_path.ACTIVE_CONNECTOR
    records = []

    def segments(owner, transfer, info, tables):
        layout = list(owner.layout(transfer))
        selected = owned_frontier_blocks(tables, layout, info['block_size'])
        valid = sum(end*info['token_bytes']*2 for depth, layer, end in layout)
        physical = len(selected)*info['block_bytes']*2
        if transfer.outgoing and info is owner.transport.info:
            records.append(dict(tid=transfer.tid, seed=transfer.request.sampling_params.seed,
                cut=transfer.cut, semantic_valid_KV_bytes=valid, physical_wire_KV_bytes=physical,
                extra_padding_bytes=physical-valid, selected_owned_blocks=len(selected),
                scope='Only this request owns these blocks. Padding and unused layer bytes remain invalid. Native import marks only the original semantic frontier.'))
        for block in selected:
            for base in (info['key_ptr'], info['value_ptr']):
                yield base+block*info['block_bytes'], info['block_bytes'], info['device']
    connector.segments = segments
    from vllm_rlt.engine.stage_engine import StageEngine
    close = StageEngine.close
    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT / 'raw' / cfg['name'] / f'owned_block_wire_{os.getpid()}.json', dict(
                role=engine.stage_config.role, records=records,
                scope='Full owned-block Native wire. No model computation, validity frontier, reservation, or ACK rule changes. This transfers extra invalid bytes, not additional valid recurrent state.'))
    StageEngine.close = closed
