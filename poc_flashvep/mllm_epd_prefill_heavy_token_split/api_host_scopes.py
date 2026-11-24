"""Read-only API CPU enclosures that distinguish preprocessing from IPC."""
from contextvars import ContextVar
import threading
import time


HTTP_REQUEST_ID=ContextVar('campaign_http_request_id',default=None)


def install(recorder):
    from vllm.renderers.base import BaseRenderer
    from vllm.v1.serial_utils import MsgpackEncoder
    originals=[]

    def observe(cls,attribute,kind):
        original=getattr(cls,attribute)
        def measured(owner,*args,**kwargs):
            if not recorder.active:return original(owner,*args,**kwargs)
            begin=time.monotonic_ns();cpu_begin=time.thread_time_ns()
            request_id=HTTP_REQUEST_ID.get()
            if request_id is None and args:request_id=getattr(args[0],'request_id',None)
            try:
                return original(owner,*args,**kwargs)
            finally:
                recorder.rows.append(dict(kind=kind,begin_ns=begin,end_ns=time.monotonic_ns(),
                    thread_cpu_ns=time.thread_time_ns()-cpu_begin,thread_id=threading.get_ident(),
                    request_id=request_id))
        originals.append((cls,attribute,original))
        setattr(cls,attribute,measured)

    observe(BaseRenderer,'_process_multimodal','multimodal_preprocess_host')
    observe(MsgpackEncoder,'encode','ipc_encode_host')
    observe(MsgpackEncoder,'encode_into','ipc_encode_into_host')
    return originals
