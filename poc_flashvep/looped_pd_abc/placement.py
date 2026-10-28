"""C1: initial placement only. A and B remain disabled in standalone comparisons."""
import time
import json
from pathlib import Path
from .common import append

def install(engine,policy,path):
 calibration_path=Path(__file__).resolve().parents[1]/'reports/looped_pd_abc/configs/service_calibration.json'
 model=engine.model_config.model_type
 if model=='huginn_raven':model='huginn'
 coefficients=json.loads(calibration_path.read_text()).get(model) if calibration_path.exists() else None
 def key(peer,incoming):
  start=time.perf_counter()
  existing=[w for w in engine.transfers.values() if getattr(w,'p' if peer.role=='prefill' else 'd')==peer.name
   and not getattr(w,'p_released' if peer.role=='prefill' else 'd_released')]
  if policy=='queue':score=len(existing)
  elif policy=='token':
   score=sum(len(w.request.prompt_token_ids) if peer.role=='prefill' else w.request.sampling_params.max_tokens-len(w.request.generated_token_ids) for w in existing)
  else:
   # Frozen pilot coefficients. These are MODEL_PREDICTION, not GPU timings.
   score=0.
   predicted_prefill=[];prefill_origins=[]
   for w in existing:
    n=len(w.request.prompt_token_ids)
    if peer.role=='prefill':
     total=max(.0001,coefficients['prefill_intercept']+coefficients['prefill_per_token']*n) if coefficients else .001*n+.010
     elapsed=max(0.,time.monotonic()-w.created-w.timings.get('prefill_started',0)) if 'prefill_started' in w.timings else 0.
     # Do not subtract elapsed service once for every queued request.
     # One GPU cannot consume all queue entries independently at C1 speed.
     remain=total
     predicted_prefill.append(total)
     if 'prefill_started' in w.timings:prefill_origins.append(w.created+w.timings['prefill_started'])
     if policy=='loop' and hasattr(w,'loop_frontiers'):
      R=engine.model_config.total_ut_steps
      progress=sum(a for a,b in w.loop_frontiers)/(max(n,1)*R)
      remain=max(.0001,total*(1-min(1.,progress)))
    else:
     per_token=(coefficients['decode_intercept']+coefficients['decode_per_context_token']*n) if coefficients else .010
     remain=max(.0001,per_token)*max(0,w.request.sampling_params.max_tokens-len(w.request.generated_token_ids))
    score+=remain
   if peer.role=='prefill' and policy=='service' and prefill_origins:
    origin=max(min(prefill_origins),getattr(peer,'last_completed_at',0.))
    score=max(.0001,sum(predicted_prefill)-max(0.,time.monotonic()-origin))
   if peer.role=='decode' and existing and coefficients:
    # Calibrated batch draining uses known token caps and observed outputs only.
    # It does not use future EOS positions.
    width=peer.client.info['max_active_requests']
    ratios=coefficients.get('observed_decode_batch_ratios',{'1':1.})
    remaining=sorted((max(0,w.request.sampling_params.max_tokens-len(w.request.generated_token_ids)),len(w.request.prompt_token_ids)) for w in existing)
    score=0.
    for start in range(0,len(remaining),width):
     wave=remaining[start:start+width];size=len(wave)
     bucket=min((int(k) for k in ratios),key=lambda b:abs(b-size))
     context=sum(n for left,n in wave)/size
     per_token=max(.0001,coefficients['decode_intercept']+coefficients['decode_per_context_token']*context)
     score+=max(left for left,n in wave)*per_token*ratios[str(bucket)]
  append(path,dict(time=time.monotonic(),request_id=incoming.request.request_id,worker=peer.name,role=peer.role,
   policy=policy,score=score,pending=len(existing),overhead_us=(time.perf_counter()-start)*1e6,
   evidence='MEASURED_ROUTE_INPUTS_WITH_PROFILED_COST_MODEL' if coefficients else 'MEASURED_ROUTE_INPUTS_WITH_UNCALIBRATED_COST_MODEL',A=False,B=False))
  return score,peer.reserved_blocks/peer.client.info['num_blocks'],peer.slots
 engine.placement_key=key
