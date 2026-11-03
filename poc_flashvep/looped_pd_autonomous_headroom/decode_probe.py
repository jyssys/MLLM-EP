"""Run small exact stage-local prompts through the existing D scheduler.

This measures additional work, not a migrated request or a new PD policy.
Native admission checks all real allocations. No cache or arithmetic is bypassed.
"""
import os
from .common import *


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt import SamplingParams
    if getattr(StageEngine,'_autonomous_decode_probe',False):return
    StageEngine._autonomous_decode_probe=True
    step=StageEngine.step;close=StageEngine.close
    folder=ROOT/'raw'/cfg['name']
    inputs=json.loads((ROOT/'data'/cfg['dataset']).read_text())[0]['input_ids'][cfg['model']][:cfg['D_probe_tokens']]
    started={};finished=set();last=[0.];origin=[None]
    def measured(self):
        if self.stage_config.role!='decode':return step(self)
        now=time.monotonic();marker=folder/'measurement_begin.json'
        if origin[0] is None and marker.exists():origin[0]=json.loads(marker.read_text())['time']
        background=[r for r in self.scheduler.requests.values() if not r.request_id.startswith('autonomous_probe_')
                    and 'RECURRENT' in str(r.phase) and 'PREFILL' not in str(r.phase)]
        limit=cfg.get('D_probe_count',4)
        if origin[0] is not None and background and len(started)<limit and now-last[0]>=cfg.get('D_probe_interval_s',1.):
            last[0]=now
            elapsed=now-origin[0];span=cfg['arrival_offsets'][-1]
            if .25*span<=elapsed<=.75*span:
                free=self.cache_manager.num_blocks-self.cache_manager.num_used_blocks
                required=self.cache_manager.required_blocks(len(inputs))
                # Keep a full background-request allocation free as a conservative
                # guard. Native reservations can still reject safely and are logged.
                margin=self.cache_manager.required_blocks(cfg['length']+cfg['cap']-1)
                if free>=required+margin:
                    rid=f'autonomous_probe_{len(started)}';seed=cfg['seed']+900000+len(started)
                    self._add_request(rid,inputs,SamplingParams(max_tokens=1,temperature=0.,seed=seed,
                        min_loops=cfg['depth'],max_loops=cfg['depth'],exit_threshold=1.,ignore_eos=True))
                    started[rid]=dict(time=now,seed=seed,input_ids=inputs,free_blocks=free,required_blocks=required)
                    append(folder/'D_probe_events.jsonl',dict(event='ENQUEUED',request_id=rid,**started[rid]))
                else:append(folder/'D_probe_events.jsonl',dict(event='REFUSED_LOCAL_MEMORY_GUARD',time=now,free_blocks=free,required_blocks=required,margin_blocks=margin))
        outputs=step(self);normal=[]
        for out in outputs:
            if out.request_id not in started:normal.append(out);continue
            if out.finished:
                assert len(out.token_ids)==1 and list(out.exit_depths)==[cfg['depth']]
                finished.add(out.request_id)
                append(folder/'D_probe_requests.jsonl',dict(request_id=out.request_id,**started[out.request_id],
                    finish=time.monotonic(),output_ids=list(out.token_ids),exit_depths=list(out.exit_depths),
                    evidence='MEASURED_NATIVE_D_LOCAL_FULL_DEPTH_PREFILL',scope='Additional independent prompt. Not a migrated continuation or serving speedup.'))
        return normal
    def closed(self):
        if self.stage_config.role=='decode':
            write(folder/'D_probe_completion.json',dict(enqueued=len(started),completed=len(finished),
                all_completed=set(started)==finished,scope='A zero count does not establish headroom.'))
        return close(self)
    StageEngine.step=measured;StageEngine.close=closed
