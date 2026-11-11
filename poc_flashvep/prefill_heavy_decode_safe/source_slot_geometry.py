"""Separate existing source transfer slots from unchanged compute slots."""
import os
from dataclasses import replace
from .common import *


def install(cfg):
    slots=cfg.get('discovery_P_transfer_slots')
    if slots is None:return
    assert type(slots) is int and slots>=cfg['P_slots']
    from ..native_loop_2d_oracle import native_path
    Connector=native_path.ACTIVE_CONNECTOR;create=Connector.create
    def created(cls,engine,*,config,notify):
        if engine.stage_config.role!='prefill':return create(engine,config=config,notify=notify)
        previous=engine.scheduler.config;assert not engine.scheduler.requests
        try:
            engine.scheduler.config=replace(previous,max_num_seqs=slots)
            connector=create(engine,config=config,notify=notify)
        finally:engine.scheduler.config=previous
        assert engine.scheduler.config.max_num_seqs==cfg['P_slots']
        assert len(connector.hidden)==slots,'Requested transfer slots were not applied'
        write(ROOT/'raw'/cfg['name']/f'source_slot_geometry_{os.getpid()}.json',dict(
            role='prefill',compute_slots=engine.scheduler.config.max_num_seqs,
            source_transfer_slots=len(connector.hidden),hidden_bytes=connector.hidden.numel()*connector.hidden.element_size(),
            scope='Only construction of the existing native transfer buffers sees the larger slot count. The scheduler restores its original compute limit before any request. KV credit, ACK, and native guards remain unchanged.'))
        return connector
    Connector.create=classmethod(created)
