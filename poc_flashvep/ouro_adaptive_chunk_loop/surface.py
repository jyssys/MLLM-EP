"""Separate diagnostic CUDA-event and clean natural-generation lanes."""
import gc
import time
import json
import os
import torch
from .common import ROOT, write, guard, digest

OBSERVER_SOURCE_SHA256=digest(__file__)

class QuantumObserver:
    def __init__(self,e,metadata):
        self.e=e
        self.metadata=metadata
        self.original=e.executor.submit
        self.rows=[]
        self.events=[]
        e.model_runner.synchronize()
        self.anchor=torch.cuda.Event(enable_timing=True)
        before=time.perf_counter()
        self.anchor.record()
        self.anchor.synchronize()
        after=time.perf_counter()
        self.origin=(before+after)/2
        self.uncertainty_s=(after-before)/2
        e.executor.submit=self.submit

    def submit(self,batch):
        from vllm_rlt.worker.executor import build_execution_batch
        descriptor=build_execution_batch(batch,4)
        runner=self.e.model_runner
        stream=runner.core_stream if descriptor.loop_count else runner.boundary_stream
        query_rows=sum(r.token_count for r in batch.requests)
        packed=descriptor.has_chunks
        graph_key=(query_rows,len(batch.requests),max(r.token_count for r in batch.requests) if packed else 1,packed)
        graph_eligible=bool(descriptor.emit_gate and runner.graphs and query_rows<=self.e.execution_config.cuda_graph_max_batch_size)
        graph_entry=runner.graphs.entries.get(graph_key) if graph_eligible else None
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        snapshots=[]
        for rid,r in self.e.scheduler.requests.items():
            snapshots.append(dict(id=rid,phase=r.phase.value,loops_done=r.loops_done,position=r.position,
                prompt_tokens=len(r.prompt_token_ids),num_prefilled=r.num_prefilled_tokens,
                num_preluded=r.num_preluded_tokens,chunk_start=r.prefill_chunk_start,
                prefill_depth_ranges=list(r.prefill_depth_ranges),
                output_tokens=len(r.generated_token_ids)))
        begin=time.perf_counter()
        start.record(stream)
        row=dict(quantum=len(self.rows),host_begin_s=begin,
            observer_schema_version=3,observer_source_sha256=OBSERVER_SOURCE_SHA256,host_load1=os.getloadavg()[0],
            pending_submission_events=len(runner.submissions.events),
            derived_graph_bucket=graph_key,graph_eligible=graph_eligible,
            graph_ready=bool(graph_entry is not None and graph_entry.graph is not None),
            spans=[dict(id=r.request.request_id,phase=batch.phase_for(r).value,start=r.token_start,
                tokens=r.token_count,depth=r.depth,group=4 if batch.phase_for(r).value=='prefill' else getattr(r,'loop_count',1) if batch.phase_for(r).value in ('recurrent','prefilling_recurrent') else 0)
                for r in batch.requests],state=snapshots,
            configured_C=self.e.scheduler.config.prefill_chunk_size,actual_total_rows=sum(r.token_count for r in batch.requests),
            graph_entries=len(runner.graphs.entries) if runner.graphs else 0,
            graph_counts={k:getattr(runner.graphs,k,0) for k in ('captures','replays','fallbacks')} if runner.graphs else {},
            actual_G=descriptor.loop_count,
            recurrent_depth_start=min((r.depth if r.depth is not None else 0 for r in batch.requests),default=0),
            context_min=min((r.token_start+r.token_count if batch.phase_for(r).value.startswith('prefill') or batch.phase_for(r).value=='recurrent' else r.request.position+1 for r in batch.requests),default=0),
            context_max=max((r.token_start+r.token_count if batch.phase_for(r).value.startswith('prefill') or batch.phase_for(r).value=='recurrent' else r.request.position+1 for r in batch.requests),default=0),
            resident_or_queued_requests=len(snapshots),actual_request_rows=len(batch.requests),
            actual_decode_rows=sum(r.token_count for r in batch.requests if batch.phase_for(r).value=='recurrent'),
            staged_bytes=runner.prefill_states.numel()*runner.prefill_states.element_size() if runner.prefill_states is not None else 0)
        ticket=self.original(batch)
        end.record(stream)
        row['host_submit_end_s']=time.perf_counter()
        row['graph_counts_end']={k:getattr(runner.graphs,k,0) for k in ('captures','replays','fallbacks')} if runner.graphs else {}
        self.rows.append(row)
        self.events.append((start,end))
        return ticket

    def finish(self,data):
        self.e.model_runner.synchronize()
        final_anchor=torch.cuda.Event(enable_timing=True)
        before=time.perf_counter()
        final_anchor.record()
        final_anchor.synchronize()
        after=time.perf_counter()
        host_elapsed_ms=1000*((before+after)/2-self.origin)
        gpu_elapsed_ms=self.anchor.elapsed_time(final_anchor)
        drift_ms=host_elapsed_ms-gpu_elapsed_ms
        uncertainty_ms=1000*(self.uncertainty_s+(after-before)/2)
        requests={r['id']:r for r in data['requests']}
        for row,(start,end) in zip(self.rows,self.events):
            start_s=self.origin+self.anchor.elapsed_time(start)/1000
            end_s=self.origin+self.anchor.elapsed_time(end)/1000
            row.update(self.metadata,device_stream_interval_ms=start.elapsed_time(end),
                aligned_device_begin_s=start_s,aligned_safe_completion_s=end_s,
                scheduling_impact_ms=1000*(max(end_s,row['host_submit_end_s'])-row['host_begin_s']),
                enqueue_to_ready_ms=1000*max(0,start_s-row['host_begin_s']),
                clock_alignment_uncertainty_ms=1000*self.uncertainty_s,
                final_anchor_host_minus_device_elapsed_ms=drift_ms,
                alignment_error_bound_ms=abs(drift_ms)+uncertainty_ms,
                evidence='DIAGNOSTIC_CUDA_EVENTS_NOT_CLEAN_SERVING',
                completed_token_residuals=[])
            for snapshot in row['state']:
                rid=snapshot['id']
                if rid not in requests or snapshot['phase'] not in ('recurrent','prelude','coda'):
                    continue
                future=next((t for t in requests[rid]['token_times'] if t>=row['host_begin_s']),None)
                if future is not None:
                    past=[t for t in requests[rid]['token_times'] if t<row['host_begin_s']]
                    row['completed_token_residuals'].append(dict(id=rid,phase=snapshot['phase'],
                        loops_done=snapshot['loops_done'],remaining_R=max(0,4-snapshot['loops_done']),
                        last_completed_token_s=past[-1] if past else None,
                        time_since_last_token_ms=1000*(row['host_begin_s']-past[-1]) if past else None,
                        time_to_completed_next_token_ms=1000*(future-row['host_begin_s']),
                        label_only_future_not_predictor_feature=True))
        self.e.executor.submit=self.original
        return self.rows

def select_cases(manifest,split,length,width,trial):
    long=[c for c in manifest['cases'] if c['split']==split and c.get('P')==length]
    short=[c for c in manifest['cases'] if c['split']==split and c['domain']=='conversation_proxy' and len(c['input_ids'])<=1024]
    assert long and len(short)>=max(1,width-1),('Unavailable frozen shape',split,length,width)
    cases=[short[(trial*(width-1)+i)%len(short)] for i in range(width-1)]
    cases.append(long[trial%len(long)])
    return cases,[0.]*(width-1)+[.250 if width>1 else 0.]

@torch.inference_mode()
def run_job(model,job,target):
    from .worker import engine,dispose,bind_legacy
    old=bind_legacy(job)
    manifest=json.loads((ROOT/'workload_manifest.json').read_text())
    for cell_i,(mode,chunk) in enumerate(job['cells']):
        length,width=job['shape']
        e=engine(model,mode,chunk,width,job)
        e.frontend_stop_token_ids=manifest['stop_ids']
        cases,arrivals=select_cases(manifest,job['split'],length,width,job['trials'][0])
        warm=old.serve(e,cases,16,arrivals,.251,observe=False)
        write(target/f'{mode}_C{chunk}_warm.json',warm)
        for trial in job['trials']:
            file=target/f'{mode}_C{chunk}_T{trial}.json'
            if file.exists():
                continue
            cases,arrivals=select_cases(manifest,job['split'],length,width,trial)
            meta=dict(run_id=job['id'],physical_gpu=guard(),split=job['split'],policy=mode,configured_C=chunk,
                P=length,N=width,trial=trial,source_families=[c['family'] for c in cases])
            torch.cuda.reset_peak_memory_stats()
            observer=QuantumObserver(e,meta) if job.get('trace',True) else None
            data=old.serve(e,cases,job.get('cap',32),arrivals,.251,drain=120,observe=False)
            if observer:
                data['quantum_observations']=observer.finish(data)
                del observer
            data.update(meta,evidence='DIAGNOSTIC_EVENT_LANE' if job.get('trace',True) else 'LIVE_MEASURED_CLEAN_NATURAL_GENERATION',
                cap=job.get('cap',32),context_scope='One long arrival amid independent short requests. Not all requests have P tokens.')
            write(file,data)
        dispose(e)
        del e,data,warm
        gc.collect()
        torch.cuda.empty_cache()
    return dict(status='COST_SURFACE_COMPLETE',split=job['split'])
