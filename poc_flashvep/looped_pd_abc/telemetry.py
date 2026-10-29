"""Nonblocking evidence capture. Disk waits never run in the request callback."""
import atexit
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

_writer=None
_lock=threading.Lock()
_original_write=None
_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_installed=False
_missing=object()

class EvidenceWriter:
 def __init__(self):
  self.queue=queue.SimpleQueue();self.error=None;self.closed=False
  self.records=0;self.enqueue_seconds=0.;self.max_enqueue_seconds=0.;self.peak_queue_records=0
  self.background_records=0;self.background_batches=0;self.background_seconds=0.
  self.thread=threading.Thread(target=self._run,name='ABC-evidence-writer',daemon=True);self.thread.start()

 def append(self,path,value):
  if self.closed or self.error:raise RuntimeError('Evidence writer is unavailable: '+str(self.error))
  begin=time.perf_counter()
  text=json.dumps(value,default=str)+'\n'
  self.queue.put((str(path),text))
  elapsed=time.perf_counter()-begin;self.records+=1;self.enqueue_seconds+=elapsed
  self.max_enqueue_seconds=max(self.max_enqueue_seconds,elapsed)
  self.peak_queue_records=max(self.peak_queue_records,self.queue.qsize())

 def _run(self):
  carry=_missing
  try:
   while True:
    item=self.queue.get() if carry is _missing else carry;carry=_missing
    if item is None:return
    if isinstance(item,threading.Event):item.set();continue
    batch=[item]
    for _ in range(127):
     try:item=self.queue.get_nowait()
     except queue.Empty:break
     if item is None or isinstance(item,threading.Event):carry=item;break
     batch.append(item)
    groups=defaultdict(list)
    for path,text in batch:groups[path].append(text)
    for name,texts in groups.items():
     begin=time.perf_counter();path=Path(name);path.parent.mkdir(parents=True,exist_ok=True)
     with path.open('a') as stream:
      stream.write(''.join(texts));stream.flush();os.fsync(stream.fileno())
     self.background_seconds+=time.perf_counter()-begin
     self.background_batches+=1;self.background_records+=len(texts)
  except BaseException as error:self.error=repr(error)

 def flush(self):
  if self.closed:return
  if self.error:raise RuntimeError('Evidence write failed: '+self.error)
  barrier=threading.Event();self.queue.put(barrier)
  while not barrier.wait(.1):
   if self.error or not self.thread.is_alive():raise RuntimeError('Evidence writer failed: '+str(self.error))
  save_summary(self)

 def close(self):
  if self.closed:return
  self.flush();self.queue.put(None);self.thread.join();self.closed=True

def protocol():
 return dict(mode='async_nonblocking',telemetry_sha256=_source_sha256,pid=os.getpid(),
  attempt_id=os.environ.get('ABC_ATTEMPT_ID'),
  foreground='JSON serialization and unbounded enqueue. No foreground flush or fsync.',
  persistence='Background batched durable writes. Flush before terminal status and process exit.',
  model_changes=False)

def save_summary(writer):
 directory=os.environ.get('ABC_RUN_DIR')
 if not directory or _original_write is None:return
 _original_write(Path(directory)/f'async_evidence_summary_{os.getpid()}.json',dict(**protocol(),
  enqueued_records=writer.records,background_written_records=writer.background_records,
  foreground_enqueue_seconds=writer.enqueue_seconds,foreground_max_enqueue_seconds=writer.max_enqueue_seconds,
  background_write_seconds=writer.background_seconds,background_fsync_batches=writer.background_batches,
  peak_queued_records=writer.peak_queue_records,error=writer.error,
  scope='Process lifetime, including warmup. Background durations are not additive request critical-path components.'))

def get_writer():
 global _writer
 if _writer is None:
  with _lock:
   if _writer is None:
    _writer=EvidenceWriter()
    directory=os.environ.get('ABC_RUN_DIR')
    if directory and _original_write is not None:
     _original_write(Path(directory)/f'async_evidence_writer_{os.getpid()}.json',protocol())
 return _writer

def append(path,value):get_writer().append(path,value)

def flush():
 if _writer is not None:_writer.flush()

def close():
 if _writer is not None:
  try:_writer.close()
  except BaseException as error:
   directory=os.environ.get('ABC_RUN_DIR')
   if directory and _original_write is not None:
    _original_write(Path(directory)/f'async_evidence_failure_{os.getpid()}.json',dict(**protocol(),error=repr(error)))
   raise

def install():
 global _installed,_original_write
 if _installed:return
 from . import common
 _installed=True;_original_write=common.write
 os.environ['ABC_ASYNC_EVIDENCE']='1'
 def write(path,value):
  path=Path(path)
  if path.name=='configuration.json' and isinstance(value,dict) and 'attempt_id' in value.get('arguments',{}):
   os.environ['ABC_RUN_DIR']=str(path.parent)
   os.environ['ABC_ATTEMPT_ID']=value['arguments']['attempt_id']
   _original_write(path.parent/'evidence_protocol.json',protocol())
  if path.name=='status.json' and isinstance(value,dict) and value.get('status') in ('COMPLETE','FAILED'):
   try:flush()
   except BaseException as error:
    failed=dict(value,status='FAILED',evidence_flush_error=repr(error))
    _original_write(path,failed);raise
  return _original_write(path,value)
 common.append=append;common.write=write
 atexit.register(close)
 # Multiprocessing uses its own finalizers before os._exit.
 from multiprocessing.util import Finalize
 Finalize(None,close,exitpriority=20)
