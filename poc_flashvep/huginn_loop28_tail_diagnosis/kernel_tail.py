"""Attempt activity-only kernel capture after every measured request reaches D."""
import ctypes,os,time
from .common import *


class KernelTail:
    def __init__(self,cfg):
        self.cfg=cfg;self.seen=set();self.started=False;self.handle=None;self.start=None
        p=(ROOT/'tools/activity_build_v2.json' if cfg.get('diagnosis_kernel_revision2') else
            REPO/'poc_flashvep/reports/looped_pd_fluidtoken_vs_dynamic_loop_split/tools/activity_build.json')
        self.build=json.loads(p.read_text())
        assert digest(self.build['library'])==self.build['library_SHA256']
        assert digest(REPO/'poc_flashvep/fluidtoken_loop_split/kernel_activity.cc')==self.build.get('included_source_SHA256',self.build['source_SHA256'])
        self.handle=ctypes.CDLL(self.build['library'])
        self.handle.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        self.handle.activity_end.argtypes=[ctypes.c_char_p]
        self.handle.activity_dropped.restype=ctypes.c_uint64
        self.handle.activity_invalid.restype=ctypes.c_uint64
        self.handle.activity_count.restype=ctypes.c_size_t
        if cfg.get('diagnosis_kernel_revision2'):
            warm=time.monotonic();status=self.handle.activity_prepare();assert status==0,status
            self.warm_initialization_ms=1000*(time.monotonic()-warm)
        self.mapping={};self.error=None

    def clock(self):
        value=ctypes.c_uint64();a=time.monotonic()
        status=self.handle.activity_timestamp(ctypes.byref(value));b=time.monotonic()
        assert status==0
        return dict(cupti_ns=value.value,host_lower=a,host_upper=b,host_midpoint=(a+b)/2)

    def observe(self,engine,batch):
        for span in batch.spans:
            seed=span.request.sampling_params.seed
            if 0<=seed-self.cfg['seed']<self.cfg['count']:self.seen.add(seed)
        if self.started or self.error or len(self.seen)<self.cfg['count']:return
        begin=time.monotonic()
        status=(self.handle.activity_resume() if self.cfg.get('diagnosis_kernel_revision2') else self.handle.activity_begin())
        self.start_call_ms=1000*(time.monotonic()-begin)
        if status:
            self.error=dict(start_status=status);return
        self.started=True;self.start=self.clock()
        try:
            from cuda.bindings import driver
            status,context=driver.cuCtxGetCurrent();assert int(status)==0
            cupti=self.handle if self.cfg.get('diagnosis_kernel_revision2') else ctypes.CDLL('/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so')
            get=cupti.cuptiGetStreamIdEx
            get.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint8,ctypes.POINTER(ctypes.c_uint32)]
            runner=engine.model_runner
            streams=dict(core=runner.core_stream,boundary=runner.boundary_stream,
                prompt=getattr(runner,'net_prompt_stream',None),
                decode_capture=getattr(runner.graphs,'capture_stream',None),
                prompt_capture=getattr(engine._autonomous_prefill_graphs,'capture_stream',None))
            for label,stream in streams.items():
                if stream is None:continue
                identifier=ctypes.c_uint32()
                error=get(int(context),int(stream.cuda_stream),0,ctypes.byref(identifier))
                self.mapping[label]=dict(cuda_stream=int(stream.cuda_stream),cupti_stream=identifier.value,status=error)
        except Exception as error:
            self.mapping['mapping_error']=repr(error)

    def close(self):
        if self.started:
            stop=self.clock();begin=time.monotonic()
            status=self.handle.activity_end(str(ROOT/'raw'/self.cfg['name']/'kernel_tail.csv').encode())
            result=dict(start=self.start,stop=stop,status=status,
                start_call_ms=self.start_call_ms,export_after_outputs_ms=1000*(time.monotonic()-begin),
                dropped=self.handle.activity_dropped(),invalid=self.handle.activity_invalid(),
                records=self.handle.activity_count(),streams=self.mapping,build=self.build)
            result['warm_initialization_ms']=getattr(self,'warm_initialization_ms',None)
        else:result=dict(status='NOT_STARTED',error=self.error,measured_requests_seen=len(self.seen))
        write(ROOT/'raw'/self.cfg['name']/'kernel_tail_capture.json',dict(result,
            scope='Actual CUPTI kernel intervals in the final imported-request tail. No API callback subscription. Export occurs after outputs. Coverage and profiler sensitivity require separate checks. No SM, HBM, or cache counters.'))


def prepare():
    initialize();names=[]
    for arm in ('zero','full'):
        cfg=json.loads((ROOT/'configs'/f'tail_{arm}_events_n64.json').read_text())
        cfg.update(name=f'tail_{arm}_kernel_tail_n64',diagnosis_kernel_tail=True,
            phase='SHORT_TAIL_ACTIVITY_ONLY_KERNEL_DIAGNOSTIC')
        save(cfg);names.append(cfg['name'])
    write(ROOT/'state/kernel_tail_plan.json',dict(jobs=names,
        scope='Optional two-cell tail capture after the original ten-cell owner closes. No concurrent GPU work. Stop a stalled capture without extending its timeout.'))
    print(' '.join(names))


if __name__=='__main__':prepare()
