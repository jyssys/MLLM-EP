"""Observe selected CPU boundaries without CUDA synchronization."""
import contextlib,gc,time,threading


class HostProbe:
    def __init__(self,enabled):
        self.enabled=enabled;self.rows=[];self.gc_starts={}
        if enabled:gc.callbacks.append(self.gc_callback)
    def gc_callback(self,phase,info):
        generation=info['generation']
        if phase=='start':self.gc_starts[generation]=(time.monotonic(),time.thread_time())
        elif generation in self.gc_starts:
            start,cpu=self.gc_starts.pop(generation)
            self.record('GC',start,cpu,generation=generation,collected=info.get('collected'))
    def record(self,kind,start,cpu,**metadata):
        end=time.monotonic();wall=1000*(end-start)
        if kind=='ADD_REQUEST' or wall>=2.:
            self.rows.append(dict(kind=kind,start=start,end=end,wall_ms=wall,
                thread_CPU_ms=1000*(time.thread_time()-cpu),thread_name=threading.current_thread().name,
                thread_id=threading.get_ident(),**metadata))
    @contextlib.contextmanager
    def span(self,kind,**metadata):
        if not self.enabled:yield;return
        start=time.monotonic();cpu=time.thread_time()
        try:yield
        finally:self.record(kind,start,cpu,**metadata)
    def close(self):
        if self.enabled:gc.callbacks.remove(self.gc_callback)
        return dict(rows=self.rows,evidence='MEASURED_HOST_BOUNDARIES',
            scope='Wall and thread CPU intervals. These intervals can overlap GPU work. They are not CUDA kernel times or additive latency percentiles.')
