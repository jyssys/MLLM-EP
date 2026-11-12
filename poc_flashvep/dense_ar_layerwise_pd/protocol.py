"""Validate the exact layer frontier before state ownership changes."""
from dataclasses import dataclass
from enum import Enum


class Phase(str,Enum):
    SOURCE='SOURCE'
    EXPORTED='EXPORTED'
    IMPORTED='IMPORTED'
    COMPLETE='COMPLETE'
    RELEASED='RELEASED'


@dataclass
class Frontier:
    request_id:str
    prompt_tokens:int
    cut:int
    total_layers:int=36
    phase:Phase=Phase.SOURCE
    prefix_valid:bool=False
    hidden_valid:bool=False
    suffix_valid:bool=False
    source_released:bool=False

    def export(self):
        assert self.phase==Phase.SOURCE and 0<self.cut<self.total_layers
        assert self.prompt_tokens>0
        self.prefix_valid=self.hidden_valid=True;self.phase=Phase.EXPORTED

    def import_state(self,*,kv_complete,hidden_complete):
        assert self.phase==Phase.EXPORTED and kv_complete and hidden_complete
        self.phase=Phase.IMPORTED

    def complete(self,start_layer,end_layer):
        assert self.phase==Phase.IMPORTED
        assert (start_layer,end_layer)==(self.cut,self.total_layers)
        self.suffix_valid=True
        self.phase=Phase.RELEASED if self.source_released else Phase.COMPLETE

    def release(self,*,native_ack):
        assert self.phase in (Phase.IMPORTED,Phase.COMPLETE) and native_ack and not self.source_released
        self.source_released=True
        # Source release does not cancel destination-owned suffix work.
        if self.phase==Phase.COMPLETE:self.phase=Phase.RELEASED

    def decode_allowed(self):
        return self.prefix_valid and self.hidden_valid and self.suffix_valid
