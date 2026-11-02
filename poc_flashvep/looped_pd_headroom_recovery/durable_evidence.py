"""Reuse the existing durable background writer, including captured aliases."""
import os
import sys
from .common import ROOT,write


def route_aliases(modules,old_append,new_append,old_write,new_write):
    changed=[]
    for name,module in modules:
        if not name.startswith('poc_flashvep.') or module is None:continue
        for field,old,new in [('append',old_append,new_append),('write',old_write,new_write)]:
            if getattr(module,field,None) is old:
                setattr(module,field,new);changed.append(name+'.'+field)
    return changed


def install(cfg):
    from ..looped_pd_abc import common as base,telemetry
    if getattr(base,'_recovery_durable_aliases',False):return
    base._recovery_durable_aliases=True
    previous_append=base.append;previous_write=base.write
    telemetry.install()
    changed=route_aliases(list(sys.modules.items()),previous_append,base.append,previous_write,base.write)
    write(ROOT/'raw'/cfg['name']/f'durable_evidence_aliases_{os.getpid()}.json',dict(pid=os.getpid(),aliases=changed,
        writer_source_SHA256=telemetry._source_sha256,evidence='RUNTIME_EVIDENCE_TRANSPORT_CONFIGURATION',
        scope='All original records remain. JSON serialization and enqueue stay in the callback. A background thread performs durable batched writes. Native model, KV, NIXL and scheduling code are unchanged.'))
