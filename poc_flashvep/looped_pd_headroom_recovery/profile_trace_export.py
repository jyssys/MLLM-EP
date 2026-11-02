"""Retain the full Chrome trace without an unused native event-table build."""
from .common import write


class ArtifactTar:
    """Keep uncompressed bytes and native verification unchanged."""
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def open(self,name=None,mode='r',*args,**kwargs):
        if mode=='w:gz':kwargs.setdefault('compresslevel',1)
        return self.original.open(name,mode,*args,**kwargs)


def exporter(original):
    def export(profiler,staging,config,device):
        if tuple(config.activities)!=('cuda',) or any((config.record_shapes,config.with_stack,config.profile_memory,config.with_flops)):
            return original(profiler,staging,config,device)
        profiler.export_chrome_trace(str(staging/'trace.json'))
        wait=getattr(profiler,'wait_for_exports',None)
        if wait is not None:wait()
        write(staging/'native_event_table_status.json',dict(status='NOT_COLLECTED',
            reason='CUDA-only device attribution uses the complete Chrome trace. Native FunctionEvent CPU-tree construction is not required.',
            retained='Full native Chrome trace, CUDA runtime events, kernel events, memcpy events, and native archive integrity checks.',
            omitted='The optional derived profiler.events() table. No CPU self-time, shape, stack, memory, or FLOP claim is made.',
            lossless_archive_gzip_level=1,
            scope='Artifact export only, after all measured responses. No collection setting or GPU execution changes.'))
    return export


def install():
    from vllm_rlt import profiling_artifacts
    if getattr(profiling_artifacts,'_recovery_trace_only_export',False):return
    profiling_artifacts.export_profile=exporter(profiling_artifacts.export_profile)
    profiling_artifacts.tarfile=ArtifactTar(profiling_artifacts.tarfile)
    profiling_artifacts._recovery_trace_only_export=True
