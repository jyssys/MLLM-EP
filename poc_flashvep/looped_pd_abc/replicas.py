"""Colocated reference using unchanged native StageClient workers, not model parallelism."""
from vllm_rlt.engine.llm_engine import StageClient
from vllm_rlt.engine.stage_config import StageConfig
import time

class ReplicaPool:
 def __init__(self,model,devices,configuration):
  self.clients=[];self.assignment={};self.worker_metrics={};self.metrics=[]
  try:
   for i in devices:
    self.clients.append(StageClient(model,device=i,stage_config=StageConfig(stage_id=f'colocated-{i}'),
     startup_timeout=300,wait_for_ready=False,**configuration))
   for c in self.clients:c.wait_ready()
  except BaseException:
   for c in self.clients:c.close()
   raise
 def add_request(self,rid,tokens,params):
  chosen=min(range(len(self.clients)),key=lambda i:len(self.clients[i].requests))
  self.clients[chosen].add_request(rid,tokens,params);self.assignment[rid]=chosen
 def step(self):
  out=[]
  for c in self.clients:
   c.poll();out.extend(c.outputs);c.outputs.clear()
  if not out:time.sleep(.0005)
  return out
 def stats(self):return {str(i):c.stats() for i,c in enumerate(self.clients)}
 def close(self):
  for i,c in enumerate(self.clients):c.close();self.worker_metrics[str(i)]=c.final_stats
