"""Publish immutable, completed KV planes through the existing Native transport."""
import time
from types import SimpleNamespace
from .progressive import delta_layout, disjoint_publication


def publish(connector, transfer, depth, event, records, mechanism):
    from ..native_loop_2d_oracle.packing import packed_segments
    from ..looped_pd_abc.segments import coalesce
    from vllm_rlt.pd.transport import partition_segments
    assert transfer.outgoing and not transfer.cancelled and not transfer.compute_done
    assert event is not None, 'Early KV requires the exact producer event'
    assert connector.cache.has_allocation(transfer.tid), 'Source allocation must remain live'
    assert connector.cache.storage_depths == connector.cfg['depth'], (
        'Immutable early publication requires disjoint recurrent KV planes')
    view = SimpleNamespace(cut=depth, request=transfer.request)
    published = getattr(transfer, 'net_published_planes', {})
    layout = delta_layout(list(connector.layout(view)), published)
    if not layout:
        return
    if connector.cfg.get('monotone_early_KV_protocol'):
        connector.notify('idle_immutable_KV_frontier', tid=transfer.tid,
            depth=depth, KV_only=True, full_prompt=len(transfer.request.prompt_token_ids))
    peer = connector.transport.peers[transfer.peer]
    local = list(packed_segments(connector.transport.info,
        connector.cache.plane_block_tables(transfer.tid), layout))
    remote = list(packed_segments(peer, transfer.target_tables, layout))
    local, remote = coalesce(local, remote)
    before = transfer.next_sequence
    for a, b, amount in partition_segments(local, remote,
            connector.config.transfer_chunk_bytes, connector.config.max_transfer_descriptors):
        transfer.chunks.append((event, transfer.next_sequence, a, b, amount))
        transfer.next_sequence += 1
    disjoint_publication(layout, published)
    transfer.net_published_planes = published
    records.append(dict(time=time.monotonic(), event='EARLY_KV_QUEUED', tid=transfer.tid,
        seed=transfer.request.sampling_params.seed, depth=depth,
        KV_bytes=sum(s[1] for s in local), fragments=transfer.next_sequence-before,
        mechanism=mechanism,
        scope='Immutable valid KV planes. Every fragment retains its producing CUDA event. Native final commit, hidden handoff, expected count, and ACK remain required.'))
