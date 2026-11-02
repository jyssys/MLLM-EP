"""Use the native supported CUDA-only profiler without replacing execution."""
import json
import os
import time
from .common import ROOT,write,append


def clock():
    lo=time.monotonic()
    epoch=time.time_ns()
    hi=time.monotonic()
    return dict(monotonic=(lo+hi)/2,epoch_ns=epoch,uncertainty_s=hi-lo)


def install(cfg):
    import torch
    from .profile_trace_export import install as install_trace_export
    install_trace_export()
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.profiling import ProfileConfig
    initialize=StageEngine.__init__
    close=StageEngine.close
    step=StageEngine._step
    folder=ROOT/'raw'/cfg['name']
    roles=cfg.get('capture_roles',[cfg.get('capture_role','prefill')])
    def initialized(self,*args,**kwargs):
        result=initialize(self,*args,**kwargs)
        role=self.stage_config.role
        if role in roles:
            self.start_profile(ProfileConfig(output_dir=str(folder/'cuda_only_profile'/role),activities=('cuda',),
                record_shapes=False,with_stack=False,profile_memory=False,with_flops=False),role=self.stage_config.role)
            anchor=clock()
            self.profiling.current.profiler.add_metadata_json('recovery_clock',json.dumps(anchor))
            write(folder/f'cuda_only_clock_{role}.json',dict(**anchor,pid=os.getpid(),role=role,
                logical_device=self.model_runner.device.index,
                evidence='MEASURED_HOST_CLOCK_PAIR_FOR_NATIVE_KINETO',
                scope='Start before worker-ready and request warm-up. Retain baseTimeNanoseconds when converting exported device timestamps.'))
            if cfg.get('native_capture_seconds'):
                self.profiling.current.profiler.toggle_collection_dynamic(False,[torch.profiler.ProfilerActivity.CUDA])
                self._recovery_cuda_window='WAIT'
        return result
    def stepped(self,*args,**kwargs):
        state=getattr(self,'_recovery_cuda_window',None)
        role=self.stage_config.role
        marker=folder/'measurement_begin.json'
        if state=='WAIT' and marker.exists():
            begin=json.loads(marker.read_text())['time']
            delay=cfg.get('native_capture_delay_s',.3*cfg['arrival_offsets'][-1])
            if time.monotonic()>=begin+delay:
                before=time.monotonic()
                self.profiling.current.profiler.toggle_collection_dynamic(True,[torch.profiler.ProfilerActivity.CUDA])
                after=time.monotonic()
                self._recovery_cuda_window='ON';self._recovery_cuda_until=after+cfg['native_capture_seconds']
                append(folder/f'cuda_only_events_{role}.jsonl',dict(event='COLLECTION_ENABLED',time=after,call_seconds=after-before,**clock()))
        elif state=='ON' and time.monotonic()>=self._recovery_cuda_until:
            end=time.monotonic()
            self.profiling.current.profiler.toggle_collection_dynamic(False,[torch.profiler.ProfilerActivity.CUDA])
            self._recovery_cuda_window='OFF'
            append(folder/f'cuda_only_events_{role}.jsonl',dict(event='COLLECTION_DISABLED',time=end,call_seconds=time.monotonic()-end,**clock()))
        return step(self,*args,**kwargs)
    def closed(self):
        role=self.stage_config.role
        if role in roles and self.profiling.recording:
            append(folder/f'cuda_only_events_{role}.jsonl',dict(event='METRIC_WINDOW_END',**clock()))
            self.stop_profile()
            status=self.wait_for_profile_artifacts(cfg.get('profile_finalization_timeout',300))
            write(folder/f'cuda_only_status_{role}.json',status)
        return close(self)
    StageEngine.__init__=initialized
    StageEngine.close=closed
    StageEngine._step=stepped


def analyze(name,clean,role='prefill'):
    import tarfile
    from .common import configure
    from .profile_analysis import union
    from ..looped_pd_atomic_prefill import evidence
    folder=ROOT/'raw'/name
    archives=list((folder/'cuda_only_profile'/role).glob('rank-*/cycle-*.tar.gz'))
    if len(archives)!=1:return dict(name=name,status='NATIVE_CUDA_TRACE_NOT_AVAILABLE')
    with tarfile.open(archives[0]) as archive:
        member=next(m for m in archive.getmembers() if m.name.endswith('/trace.json') or m.name=='trace.json')
        raw=json.load(archive.extractfile(member))
    anchor=json.loads((folder/f'cuda_only_clock_{role}.json').read_text())
    base=raw.get('baseTimeNanoseconds')
    assert isinstance(base,int),'Native Chrome trace must retain its absolute timestamp base'
    assert raw['recovery_clock']=={k:anchor[k] for k in ('monotonic','epoch_ns','uncertainty_s')}
    window=json.loads((folder/'window.json').read_text())
    cohort_duration=window['end']-window['start']
    events=[json.loads(s) for s in (folder/f'cuda_only_events_{role}.jsonl').read_text().splitlines()]
    enabled=next((e for e in events if e['event']=='COLLECTION_ENABLED'),None)
    disabled=next((e for e in events if e['event']=='COLLECTION_DISABLED'),None)
    if enabled:
        assert disabled,'Bounded capture did not close its observation window'
        window=dict(window,start=max(window['start'],enabled['time']),end=min(window['end'],disabled['time']))
    kernels=[]
    for event in raw['traceEvents']:
        if event.get('ph')!='X' or event.get('cat')!='kernel':continue
        if 'device' in event.get('args',{}) and event['args']['device']!=anchor['logical_device']:continue
        start=anchor['monotonic']+(base+1000*event['ts']-anchor['epoch_ns'])/1e9
        end=start+event['dur']/1e6
        if end<window['start'] or start>window['end']:continue
        kernels.append(dict(start_monotonic=start,end_monotonic=end,name=event['name'],pid=anchor['pid']))
    if not kernels:
        result=dict(name=name,clean=clean,role=role,status='EMPTY_CUDA_ACTIVITY_TRACE',
            kernel_busy_fraction=None,eligible_for_matched_clean=False,
            evidence='NOT_MEASURED_GPU_ACTIVITY',scope='Inference can complete while the requested activity collector returns no kernels. This is not zero GPU busy. Do not invent a clock offset.')
        write(folder/f'profile_analysis_{role}.json',result)
        if role=='prefill':write(folder/'profile_analysis.json',result)
        return result
    duration=window['end']-window['start']
    intervals=[(max(window['start'],r['start_monotonic']),min(window['end'],r['end_monotonic'])) for r in kernels]
    reference=json.loads((ROOT/'raw'/clean/'window.json').read_text())
    difference=cohort_duration/(reference['end']-reference['start'])-1
    configure();evidence.ROOT=ROOT
    parity=evidence.compare(clean,name)['passed']
    alignment=dict(anchor=anchor,window_start=window['start'],window_end=window['end'],
        kernels=kernels,evidence='MEASURED_NATIVE_CUDA_ONLY_KINETO_KERNEL_INTERVALS')
    write(folder/f'aligned_kernels_{role}.json',alignment)
    result=dict(name=name,clean=clean,role=role,status='PARSED',recorded_kernel_invocations=len(kernels),
        window_start_monotonic=window['start'],window_end_monotonic=window['end'],window_seconds=duration,
        kernel_busy_fraction=union(intervals)/duration,anchor_uncertainty_s=anchor['uncertainty_s'],
        cohort_wall_seconds=cohort_duration,bounded_capture=enabled is not None,
        matched_wall_difference_fraction=difference,exact_output_parity=parity,
        eligible_for_matched_clean=abs(difference)<=.05 and parity,
        evidence='MEASURED_NATIVE_CUDA_ONLY_KINETO',scope='Native supported profiler with CUDA activities only. No CPU operator recording. Kernel union is not SM occupancy. Reject material perturbation.')
    write(folder/f'profile_analysis_{role}.json',result)
    if role=='prefill':
        write(folder/'aligned_kernels.json',alignment)
        write(folder/'profile_analysis.json',result)
    return result
