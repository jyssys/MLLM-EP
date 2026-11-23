"""Observe coordinator stalls and optionally defer evidence-file writes."""
import gc
import json
import resource
import time
from collections import defaultdict

SELECTIVE={'credit_events.jsonl','P_wait_segments.jsonl','request_submissions.jsonl','source_retirement_notifications.jsonl'}


class Probe:
    def __init__(self,cfg,folder):
        self.folder=folder;self.trace=bool(cfg.get('coordinator_trace'))
        self.deferred=bool(cfg.get('buffer_coordinator_evidence'))
        self.lines=defaultdict(list);self.bytes=0;self.rows=[];self.gc_rows=[];self.loop_rows=[];self.send_rows=[]
        self.last_loop=None;self.last_CPU=None;self.last_thread_CPU=None;self.closed=False
        self.receive_rows=[];self.poll_rows=[]
        self.defer_gc=bool(cfg.get('defer_coordinator_cyclic_gc'))
        self.gc_originally_enabled=None;self.measurement_rows=[]
        if self.trace:gc.callbacks.append(self.gc_event)

    def begin_measurement(self):
        assert self.gc_originally_enabled is None
        self.gc_originally_enabled=gc.isenabled()
        if self.defer_gc:gc.disable()
        self.measurement_rows.append(dict(event='BEGIN',time=time.monotonic(),
            GC_enabled=gc.isenabled(),max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    def end_measurement(self):
        if self.gc_originally_enabled is None:return
        self.measurement_rows.append(dict(event='END',time=time.monotonic(),
            GC_enabled=gc.isenabled(),max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if self.gc_originally_enabled:gc.enable()
        else:gc.disable()
        self.gc_originally_enabled=None

    def gc_event(self,phase,info):
        self.gc_rows.append(dict(phase=phase,time=time.monotonic(),**info))

    def loop(self,now):
        cpu=time.process_time() if self.trace else None;thread_cpu=time.thread_time() if self.trace else None
        if self.trace and self.last_loop is not None and now-self.last_loop>.01:
            self.loop_rows.append(dict(left=self.last_loop,right=now,gap_ms=1000*(now-self.last_loop),
                process_CPU_ms=1000*(cpu-self.last_CPU),main_thread_CPU_ms=1000*(thread_cpu-self.last_thread_CPU)))
        self.last_loop=now;self.last_CPU=cpu;self.last_thread_CPU=thread_cpu

    def append(self,path,row,selective):
        start=time.monotonic() if self.trace else None
        if path.name in SELECTIVE:
            selective.setdefault(path,[]).append(row)
            return
        line=json.dumps(row,ensure_ascii=False)+'\n'
        serialized=time.monotonic() if self.trace else None
        if self.deferred:
            # Serialize now,as in the original. Do not retain a mutable row.
            self.lines[path].append(line);self.bytes+=len(line.encode())
            assert self.bytes<=256*1024**2,'Do not flush into the measured loop after exceeding the frozen evidence budget'
        else:
            with path.open('a') as stream:stream.write(line)
        if self.trace:self.rows.append(dict(file=path.name,request_id=row.get('request_id'),start=start,
            serialization_end=serialized,end=time.monotonic(),deferred=self.deferred))

    def channel(self,channel,role):
        if not self.trace:return channel
        probe=self
        class Observed:
            def __getattr__(self,name):return getattr(channel,name)
            def recv(self):
                start=time.monotonic();value=channel.recv()
                probe.receive_rows.append(dict(role=role,kind=value.get('kind'),request_id=value.get('request_id'),start=start,end=time.monotonic()))
                return value
            def poll(self,*args,**kwargs):
                start=time.monotonic();value=channel.poll(*args,**kwargs);end=time.monotonic()
                if end-start>.005:probe.poll_rows.append(dict(role=role,start=start,end=end))
                return value
            def send(self,value):
                start=time.monotonic()
                try:return channel.send(value)
                finally:probe.send_rows.append(dict(role=role,kind=value.get('kind'),request_id=value.get('request_id'),
                    start=start,end=time.monotonic(),scope='Existing sender thread serialization/socket enclosure. Not coordinator blocking or wire time.'))
        return Observed()

    def close(self):
        if self.closed:return
        self.closed=True
        self.end_measurement()
        if self.trace:gc.callbacks.remove(self.gc_event)
        flush=[]
        for path,lines in self.lines.items():
            before=time.monotonic()
            with path.open('a') as stream:stream.writelines(lines)
            flush.append(dict(file=path.name,lines=len(lines),start=before,end=time.monotonic()))
        from poc_flashvep.dense_ar_longbench400_pd.common import write
        write(self.folder/'coordinator_probe.json',dict(deferred=self.deferred,traced=self.trace,
            buffered_bytes=self.bytes,append_rows=self.rows,gc_rows=self.gc_rows,loop_gaps=self.loop_rows,
            existing_sender_enclosures=self.send_rows,post_measurement_flush=flush,
            receive_enclosures=self.receive_rows,slow_poll_enclosures=self.poll_rows,
            cyclic_GC_deferred=self.defer_gc,measurement_boundaries=self.measurement_rows,
            scope='Client timestamps are real coordinator receive times. No virtual clock or token timestamp correction. '
                'The optional intervention serializes each evidence row at the original call. Only its file write is deferred. '
                'Runtime KV/hidden metadata,ACKs,worker logging and model execution are unchanged. '
                'Optional cyclic-GC deferral affects only this coordinator process between measurement boundaries. '
                'Reference counting remains enabled. The original cyclic-GC state is restored after measurement or in finally. '
                'Normal Python finally flushes records. Hard process death before finally can lose buffered evidence. '
                'Host enclosures can overlap GC and scheduling. They are not exclusive CPU or GPU durations.'))


def install(cfg,folder):return Probe(cfg,folder)
