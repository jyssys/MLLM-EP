"""Unprofiled native/PD serving. EOS stays enabled. Raw output IDs remain available."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
import random
import uuid
import zipfile
import subprocess
from .common import ROOT,MODELS,MODEL_PINS,RLT_REVISION,UUIDS,setup,write,append,digest,append_cost

def prompts(model,count,length):
 from transformers import AutoTokenizer,PreTrainedTokenizerFast
 tokenizer=(PreTrainedTokenizerFast if model=='dory' else AutoTokenizer).from_pretrained(MODELS[model],local_files_only=True,trust_remote_code=True)
 if length:
  base=tokenizer.encode('A library contains shelves of books. Explain careful counting and clear reasoning. '*5000,add_special_tokens=False)
  return [base[i*7:i*7+length] for i in range(count)]
 source=Path('/home/esjung/MLLM-EP-github/poc_flashvep/results/refinement_aware_ep_expert_allocation/manifests/validation.jsonl')
 rows=[json.loads(line) for line in source.read_text().splitlines()][:count]
 results=[]
 for r in rows:
  encoded=tokenizer.apply_chat_template([{'role':'user','content':r['question']}],tokenize=True,add_generation_prompt=True,return_dict=False)
  results.append(list(encoded['input_ids'] if isinstance(encoded,dict) else encoded))
 return results

def configuration(model,width,chunk,graphs=False,asynchronous=False):
 from vllm_rlt import CacheConfig,SchedulerConfig,ExecutionConfig,ExitConfig
 return dict(cache_config=CacheConfig(num_blocks=4096 if model=='ouro' else 12288,block_size=16,layout='last_exited'),
  scheduler_config=SchedulerConfig(max_num_seqs=width,max_num_batched_tokens=max(2048,chunk),prefill_chunk_size=chunk,mode='refill'),
  execution_config=ExecutionConfig(async_scheduling=asynchronous,multi_stream=True,static_buffers=True,cuda_graphs=graphs,cuda_graph_max_batch_size=max(width,1),cuda_graph_max_graphs=16),
  exit_config=ExitConfig(mode='ouro_delayed' if asynchronous else 'ouro'),attention_backend='flash_attn_4')

def serve(engine,inputs,args,run,warmup=False):
 from vllm_rlt import SamplingParams
 pending=list(enumerate(inputs));active={};final={};traces=[];finished=0
 start=time.monotonic();first={};last={};inter={};arrivals={};admitted={};nextarrival=start
 rng=random.Random(20261029+run)
 while pending or active:
  now=time.monotonic()
  while pending and (args.rate>0 or len(active)<args.concurrency) and now>=nextarrival:
   index,tokens=pending.pop(0);rid=f'r{run}_q{index}'
   params=SamplingParams(max_tokens=args.cap,min_loops=MODEL_PINS[args.model][2],max_loops=MODEL_PINS[args.model][2],exit_threshold=1.,temperature=0.,seed=20261029+index)
   scheduled=nextarrival if args.rate else time.monotonic()
   engine.add_request(rid,tokens,params);active[rid]=index;arrivals[rid]=scheduled;admitted[rid]=time.monotonic();inter[rid]=[]
   nextarrival=nextarrival+rng.expovariate(args.rate) if args.rate else start
   now=time.monotonic()
  for output in engine.step():
   rid=output.request_id;now=time.monotonic();count=len(output.token_ids)
   previous=last.get(rid,(arrivals[rid],0))
   if count>previous[1]:
    first.setdefault(rid,now)
    inter[rid].append(dict(time=now,previous_time=previous[0],new_tokens=count-previous[1],first=previous[1]==0))
    last[rid]=(now,count)
   if output.finished:
    if any(depth!=MODEL_PINS[args.model][2] for depth in output.exit_depths):
     raise RuntimeError('Static-depth trace contains a non-fixed exit')
    index=active.pop(rid);finished+=1
    result=dict(model=args.model,model_revision=MODEL_PINS[args.model][1],engine_revision=RLT_REVISION,
     run=run,attempt_id=args.attempt_id,method=args.mode,sample_id=index,input_ids=inputs[index],output_ids=list(output.token_ids),
     exit_depths=list(output.exit_depths),finish_reason=str(output.finish_reason),
     input_tokens=len(inputs[index]),output_tokens=count,arrival=arrivals[rid],finish=now,
     admission=admitted[rid],harness_delivery_ms=1000*(admitted[rid]-arrivals[rid]),
     ttft_ms=1000*(first.get(rid,now)-arrivals[rid]),e2e_ms=1000*(now-arrivals[rid]),
     tpot_ms=1000*(now-first.get(rid,now))/max(count-1,1),token_events=inter[rid],
     gpus=args.gpus,prefill_devices=args.prefill,chunk_size=args.chunk,concurrency=args.concurrency,
     depth=MODEL_PINS[args.model][2],kv_layout='last_exited',dtype='bfloat16',
     evidence='MEASURED_NATURAL_GENERATION',timing_scope='Scheduled Poisson arrival to final output, including harness delivery and queue.' if args.rate else 'Harness admission to final output. No HTTP transport.',
     arrival_rate=args.rate,arrival_pattern='Poisson' if args.rate else 'closed_loop',
     truncated=str(output.finish_reason)=='length',
     lane=os.environ.get('ABC_LANE','parallel_discovery'),
     huginn_reference_compat=os.environ.get('ABC_HUGINN_REFERENCE')=='1',
     dory_adapter='packed_GEMM_native_vLLM_RoPE_v2' if args.model=='dory' else None,
     prompt_source='synthetic_library' if args.length else 'local_GSM8K_train_validation',
     seed=20261029+index,warmup=warmup,graphs=args.graphs,asynchronous=args.asynchronous,
     enable_recurrence_split=args.enable_recurrence_split,split_cut=args.split_cut if args.enable_recurrence_split else MODEL_PINS[args.model][2],enable_frontier_handoff=args.enable_frontier_handoff,
     placement=args.placement)
    final[index]=result
    append(ROOT/'raw'/args.name/'requests.jsonl',result)
    print('DONE',args.name,index,count,output.finish_reason,round(result['e2e_ms'],2),flush=True)
  if not active and pending and time.monotonic()<nextarrival:time.sleep(min(.001,nextarrival-time.monotonic()))
  if time.monotonic()-start>args.timeout:raise TimeoutError('Serving cohort exceeded watchdog')
 return final

def run(args):
 if args.mode!='pd' and (args.enable_recurrence_split or args.enable_frontier_handoff or args.placement!='upstream'):
  raise ValueError('A/B/C flags require actual PD workers')
 if (args.enable_recurrence_split or args.enable_frontier_handoff) and args.placement!='upstream':
  raise ValueError('Combined A/C or B/C policies are not qualified')
 if args.enable_frontier_handoff and not args.wavefront:
  raise ValueError('B requires its matched staged G1 control')
 setup();dest=ROOT/'raw'/args.name;dest.mkdir(parents=True,exist_ok=True)
 archive=dest/('source-'+args.attempt_id+'.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for source in Path(__file__).parent.glob('*.py'):z.write(source,'poc_flashvep/looped_pd_abc/'+source.name)
  from .common import RLT
  z.writestr('rlt_changes.patch',subprocess.check_output(['git','diff','--binary'],cwd=RLT))
 write(dest/'configuration.json',dict(arguments=vars(args),model_pin=MODEL_PINS[args.model],source_sha256=digest(__file__),visible=os.environ.get('CUDA_VISIBLE_DEVICES')))
 inputs=prompts(args.model,args.requests,args.length)
 cfg=configuration(args.model,args.concurrency,args.chunk,args.graphs,args.asynchronous)
 if args.scheduler_mode!='refill':
  from dataclasses import replace
  cfg['scheduler_config']=replace(cfg['scheduler_config'],mode=args.scheduler_mode)
 if args.wavefront:
  from dataclasses import replace
  cfg['scheduler_config']=replace(cfg['scheduler_config'],wavefront_chunked_prefill=True)
 os.environ['ABC_RUN_DIR']=str(dest)
 if args.profile:os.environ['ABC_PROFILE_RANGES']='1'
 if args.model=='dory':
  from .dory_adapter import install
  install()
 if args.model in ('huginn','dory') and (args.wavefront or args.scheduler_mode=='mixed'):
  os.environ['ABC_ALLOW_STAGED_BOUNDARY']='1'
 torch.set_num_threads(1)
 if args.mode in ('native','decode_local'):
  device_index=0 if args.mode=='native' else len(args.gpus.split(','))-1
  torch.cuda.set_device(device_index)
  physical=int(args.gpus.split(',')[device_index])
  assert str(torch.cuda.get_device_properties(device_index).uuid).removeprefix('GPU-')==UUIDS[physical].removeprefix('GPU-')
  from vllm_rlt.models import AutoModelForCausalLM
  from vllm_rlt.engine.stage_engine import StageEngine
  model=AutoModelForCausalLM.from_pretrained(MODELS[args.model],device=f'cuda:{device_index}',dtype=torch.bfloat16)
  engine=StageEngine(model,**cfg)
 elif args.mode=='dp':
  from .replicas import ReplicaPool
  engine=ReplicaPool(MODELS[args.model],range(len(args.gpus.split(','))),cfg)
 else:
  if args.enable_recurrence_split:
   if args.enable_frontier_handoff:raise ValueError('A/B combined mode is not qualified')
   os.environ['ABC_SPLIT_CUT']=str(args.split_cut)
   os.environ['ABC_ALLOW_STAGED_BOUNDARY']='1'
   if args.model=='huginn':os.environ['ABC_ALLOW_BOUNDARY_PD']='1'
   import vllm_rlt.pd.engine as pd_module
   from .split import SplitConnector
   pd_module.PDStageConnector=SplitConnector
  elif args.enable_boundary_adapter and not args.enable_frontier_handoff and args.placement=='upstream':
   os.environ['ABC_ALLOW_BOUNDARY_PD']='1'
   import vllm_rlt.pd.engine as pd_module
   from .boundary import BoundaryConnector
   pd_module.PDStageConnector=BoundaryConnector
  elif args.enable_frontier_handoff or args.placement!='upstream':
   import vllm_rlt.pd.engine as pd_module
   from .frontier import FrontierConnector,ProgressConnector,BoundaryProgressConnector
   if args.model=='huginn':os.environ['ABC_ALLOW_BOUNDARY_PD']='1'
   pd_module.PDStageConnector=FrontierConnector if args.enable_frontier_handoff else BoundaryProgressConnector if args.enable_boundary_adapter else ProgressConnector
  from vllm_rlt.pd.engine import PDEngine
  from vllm_rlt.pd import PDConfig
  devices=args.gpus.split(',');p=args.prefill
  engine=PDEngine(MODELS[args.model],pd_config=PDConfig(prefill_devices=tuple(range(p)),decode_devices=tuple(range(p,len(devices))),startup_timeout=300),
   prefill_cache_config=cfg['cache_config'],decode_cache_config=cfg['cache_config'],
   prefill_scheduler_config=cfg['scheduler_config'],decode_scheduler_config=cfg['scheduler_config'],
   execution_config=cfg['execution_config'],exit_config=cfg['exit_config'],attention_backend=cfg['attention_backend'],seed=20261029)
  if args.placement!='upstream':
   from .placement import install
   install(engine,args.placement,ROOT/'raw'/args.name/'placements.jsonl')
 if args.mode=='pd':
  physical=list(map(int,args.gpus.split(',')));mapping={}
  for name,peer in engine.peers.items():
   visible=peer.client.info['info']['device'];actual=peer.client.info['cuda_uuid']
   assert actual.removeprefix('GPU-')==UUIDS[physical[visible]].removeprefix('GPU-')
   mapping[name]=dict(role=peer.role,visible_device=visible,physical_GPU=physical[visible],uuid=actual,
    fingerprint=peer.client.info['fingerprint'],cache_geometry={k:peer.client.info['info'][k] for k in ('num_layers','depths','block_size','num_blocks')})
  write(dest/'worker_map.json',mapping)
 try:
  if args.warmup:serve(engine,inputs[:min(args.concurrency,len(inputs))],args,-1,warmup=True)
  if args.mode=='pd':
   while engine.has_unfinished_requests():engine.step()
  before={name:peer.client.stats() for name,peer in engine.peers.items()} if args.mode=='pd' else engine.stats()
  write(dest/'before_measured.json',before)
  write(dest/'coordinator_IO_before.json',append_cost())
  if args.barrier:
   barrier=ROOT/'state'/args.barrier;barrier.mkdir(parents=True,exist_ok=True)
   write(barrier/(args.gpus+'.json'),dict(pid=os.getpid(),time=time.monotonic()))
   deadline=time.monotonic()+300
   while len(list(barrier.glob('*.json')))<args.barrier_count:
    if time.monotonic()>deadline:raise TimeoutError('Contention barrier timed out')
    time.sleep(.01)
  if args.profile:
   from vllm_rlt.profiling import ProfileConfig
   engine.start_profile(ProfileConfig(output_dir=str(dest/'profile'),record_shapes=True))
  os.environ['ABC_MEASURED_ACTIVE']='1'
  for repeat in range(args.repeats):serve(engine,inputs,args,repeat)
  os.environ['ABC_MEASURED_ACTIVE']='0'
  if args.profile:
   engine.stop_profile();engine.wait_for_profile_artifacts(180)
  if args.mode=='pd':
   while engine.has_unfinished_requests():engine.step()
  after={name:peer.client.stats() for name,peer in engine.peers.items()} if args.mode=='pd' else engine.stats()
  write(dest/'after_measured.json',after)
  write(dest/'coordinator_IO_after.json',append_cost())
  write(dest/'stats.json',dict(stats=engine.stats() if hasattr(engine,'stats') else None,
    worker_metrics=getattr(engine,'worker_metrics',None),metrics=list(getattr(engine,'metrics',[]))))
 finally:
  engine.close()
  write(dest/'closed_stats.json',dict(worker_metrics=getattr(engine,'worker_metrics',None),
   metrics=list(getattr(engine,'metrics',[])),
   local_peak_allocated=torch.cuda.max_memory_allocated(device_index) if args.mode in ('native','decode_local') else 'WORKER_MEASUREMENT_REQUIRED'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=tuple(MODELS),required=True)
 p.add_argument('--mode',choices=('native','pd','dp','decode_local'),required=True);p.add_argument('--gpus',required=True);p.add_argument('--name',required=True)
 p.add_argument('--requests',type=int,default=4);p.add_argument('--length',type=int,default=128)
 p.add_argument('--cap',type=int,default=8);p.add_argument('--chunk',type=int,default=2048);p.add_argument('--concurrency',type=int,default=1)
 p.add_argument('--prefill',type=int,default=1);p.add_argument('--repeats',type=int,default=1);p.add_argument('--rate',type=float,default=0)
 p.add_argument('--timeout',type=float,default=1200);p.add_argument('--graphs',action='store_true')
 p.add_argument('--warmup',action='store_true');p.add_argument('--asynchronous',action='store_true')
 p.add_argument('--profile',action='store_true');p.add_argument('--wavefront',action='store_true')
 p.add_argument('--enable-frontier-handoff',action='store_true')
 p.add_argument('--enable-boundary-adapter',action='store_true')
 p.add_argument('--enable-recurrence-split',action='store_true');p.add_argument('--split-cut',type=int,default=1)
 p.add_argument('--placement',choices=('upstream','queue','token','service','loop'),default='upstream')
 p.add_argument('--barrier');p.add_argument('--barrier-count',type=int,default=8)
 p.add_argument('--scheduler-mode',choices=('refill','no_refill','mixed'),default='refill')
 args=p.parse_args()
 args.attempt_id=uuid.uuid4().hex
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpus
 status=ROOT/'raw'/args.name/'status.json';write(status,dict(status='RUNNING',pid=os.getpid(),time=time.time()))
 from .nvml_watch import GPUWatch
 monitor=GPUWatch(status.parent,list(map(int,args.gpus.split(','))),args.attempt_id);monitor.start()
 try:run(args)
 except BaseException as error:
  write(status,dict(status='FAILED',error=repr(error),time=time.time()));raise
 finally:monitor.stop()
 write(status,dict(status='COMPLETE',time=time.time()))

if __name__=='__main__':main()
