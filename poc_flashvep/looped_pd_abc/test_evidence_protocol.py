"""Regression at the actual worker logging seam. This file never starts a GPU model."""
import json
import os
import threading

def test_worker_append_does_not_fsync_on_the_request_thread(monkeypatch,tmp_path):
 from .common import write
 from .worker import append
 producer=threading.get_ident();foreground=[];original=os.fsync
 def tracked(descriptor):
  if threading.get_ident()==producer:foreground.append(descriptor)
  return original(descriptor)
 monkeypatch.setattr(os,'fsync',tracked)
 write(tmp_path/'configuration.json',{'arguments':{'attempt_id':'unit_evidence'}})
 append(tmp_path/'requests.jsonl',{'sample_id':1,'output_ids':[2,3]})
 assert foreground==[], 'Evidence append blocks the request thread on fsync'
 write(tmp_path/'status.json',{'status':'COMPLETE'})
 assert json.loads((tmp_path/'requests.jsonl').read_text())['output_ids']==[2,3]
