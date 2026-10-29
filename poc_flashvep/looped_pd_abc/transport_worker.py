"""Generic NIXL pacing ablation. Model operations and loop scheduling stay unchanged."""
from functools import partial
import os
from .common import ROOT,write

def main():
 import vllm_rlt.pd as pd
 from . import worker
 name=os.environ['ABC_TRANSPORT_RUN'];chunk=int(os.environ['ABC_TRANSPORT_CHUNK_MIB'])*1024**2
 inflight=int(os.environ['ABC_TRANSPORT_INFLIGHT_MIB'])*1024**2
 if not 0<chunk<=inflight:raise ValueError('Invalid generic NIXL pacing configuration')
 write(ROOT/'raw'/name/'transport_provenance.json',dict(transfer_chunk_bytes=chunk,max_inflight_bytes=inflight,
  mechanism='Generic transport pacing. No loop-frontier knowledge is added to the control.',
  source='Explicit PDConfig fields. No model, scheduler, attention, or cache-value changes.'))
 original=pd.PDConfig;pd.PDConfig=partial(original,transfer_chunk_bytes=chunk,max_inflight_bytes=inflight)
 try:worker.main()
 finally:pd.PDConfig=original

if __name__=='__main__':main()
