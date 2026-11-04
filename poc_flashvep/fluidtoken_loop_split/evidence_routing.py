"""Route campaign-specific append aliases through the existing durable writer."""
import json
import os
import sys
import time
from .common import ROOT, write


def enqueue(writer, path, value):
    if writer.closed or writer.error:
        raise RuntimeError('Evidence writer is unavailable: ' + str(writer.error))
    begin = time.perf_counter()
    # Preserve strict JSON and snapshot mutable records before asynchronous I/O.
    text = json.dumps(value, allow_nan=False) + '\n'
    writer.queue.put((str(path), text))
    elapsed = time.perf_counter() - begin
    writer.records += 1
    writer.enqueue_seconds += elapsed
    writer.max_enqueue_seconds = max(writer.max_enqueue_seconds, elapsed)
    writer.peak_queue_records = max(writer.peak_queue_records, writer.queue.qsize())


def route_aliases(modules, originals, replacement):
    changed = []
    for name, module in modules:
        if not name.startswith('poc_flashvep.') or module is None:
            continue
        if any(getattr(module, 'append', None) is value for value in originals):
            module.append = replacement
            changed.append(name + '.append')
    return changed


def install(cfg):
    if not cfg.get('route_campaign_evidence_async'):
        return
    if not cfg.get('async_evidence'):
        raise ValueError('The existing durable evidence writer must be enabled')
    from ..looped_pd_abc import telemetry
    from ..looped_pd_headroom_recovery import common as recovery
    from ..looped_pd_autonomous_headroom import common as autonomous
    originals = (recovery.append, autonomous.append)
    def append(path, value):
        enqueue(telemetry.get_writer(), path, value)
    changed = route_aliases(list(sys.modules.items()), originals, append)
    write(ROOT / 'raw' / cfg['name'] / f'campaign_evidence_routing_{os.getpid()}.json', dict(
        aliases=changed, pid=os.getpid(),
        scope='Strict JSON serialization stays on the caller. The existing durable writer performs file I/O. Model and transport execution are unchanged.'))
