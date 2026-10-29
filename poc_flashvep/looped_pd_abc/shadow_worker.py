"""Capture legal route inputs and alternate scores. Never use this run as clean latency."""
from copy import deepcopy
import os
import time
from types import SimpleNamespace
from .common import ROOT,write,append

def main():
 from . import placement
 original_install=placement.install
 name=os.environ['ABC_MANIFEST_RUN'];folder=ROOT/'raw'/name
 write(folder/'shadow_provenance.json',dict(diagnostic_only=True,
  scope='Live legal routing states with alternate cost scores. Extra CPU logging invalidates clean serving comparisons.',
  outputs='Real greedy generation, not frozen token replay',future_information_used=False))
 def install(engine,policy,path):
  original_install(engine,policy,path);actual=engine.placement_key;alternatives={}
  for alternative in ('queue','token','service','loop'):
   view=SimpleNamespace(model_config=engine.model_config,transfers=engine.transfers)
   original_install(view,alternative,folder/'shadow'/alternative/'scores.jsonl')
   alternatives[alternative]=view.placement_key
  def key(peer,incoming):
   state=[]
   for w in engine.transfers.values():
    if getattr(w,'p' if peer.role=='prefill' else 'd')!=peer.name:continue
    if getattr(w,'p_released' if peer.role=='prefill' else 'd_released'):continue
    state.append(dict(request_id=w.request.request_id,created=w.created,
     input_tokens=len(w.request.prompt_token_ids),observed_outputs=len(w.request.generated_token_ids),
     output_cap=w.request.sampling_params.max_tokens,timings=dict(w.timings),
     completed_prefill_frontiers=deepcopy(getattr(w,'loop_frontiers',None))))
   begin=time.monotonic();scores={label:callback(peer,incoming) for label,callback in alternatives.items()}
   result=actual(peer,incoming)
   append(folder/'shadow_states.jsonl',dict(time=begin,request_id=incoming.request.request_id,
    worker=peer.name,role=peer.role,reserved_blocks=peer.reserved_blocks,slots=peer.slots,
    states=state,scores=scores,actual_score=result,actual_policy=policy,
    evidence='MEASURED_CURRENT_ROUTING_STATE_WITH_MODEL_PREDICTION_SCORES',
    limitation='One diagnostic start. CPU instrumentation changes scheduling. Alternate scores are not counterfactual GPU completion times.'))
   return result
  engine.placement_key=key
 placement.install=install
 from .workload_worker import main as workload_main
 workload_main()

if __name__=='__main__':main()
