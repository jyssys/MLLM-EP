"""Optional probe-only coordinator. Extra work never counts as task completion."""
import argparse
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from . import idle_admission,proxy
from .common import ROOT,append,write,digest


class ProbeAdmission(idle_admission.IdleAdmission):
    async def try_probe(self,request_id):
        async with self.condition:
            if self.failure is not None:raise RuntimeError('D state is uncertain. No further admission is permitted.')
            assert request_id not in self.active and request_id not in self.pending
            before=self.state()
            if self.active or self.prefill_owner is not None or self.pending:
                return dict(accepted=False,before=before,reason='D stream or higher-priority admission is pending')
            self.active[request_id]=True;self.prefill_owner=request_id
            event=dict(request_id=request_id,extra_prefill=True,begin=time.monotonic(),
                admitted=time.monotonic(),before=before,after=self.state(),interference_probe=True)
            self.events.append(event)
            return dict(accepted=True,admission=event)


def application(name,encoder,prefill,decoder):
    gate=ProbeAdmission()
    # This executable owns one app in one process. Capture the original
    # coordinator's admission instance without modifying its request handlers.
    factory=idle_admission.IdleAdmission
    idle_admission.IdleAdmission=lambda:gate
    try:
        app=proxy.application('epd',encoder,prefill,[decoder],name,
            processor_handoff='grid_only',release_encoder_ipc=True,token_admission='decode_idle')
    finally:idle_admission.IdleAdmission=factory
    native_state=next(route.endpoint for route in app.routes if getattr(route,'path',None)=='/campaign_state')
    leases={};failure=None

    @app.post('/campaign/probe/acquire')
    async def acquire(request:Request):
        nonlocal failure
        body=await request.json();arm=body['arm'];assert arm in ['E','D']
        state=await native_state();before=dict(time=time.monotonic(),arm=arm,native_state=state)
        if failure is not None:return JSONResponse(dict(error='Previous probe completion is uncertain'),status_code=503)
        if leases or state['prefill_pending']==0:
            result=dict(accepted=False,reason='Probe active or no P HTTP work',before=before)
        elif arm=='E' and state['encoder_pending']:
            result=dict(accepted=False,reason='Vision encoding has priority',before=before)
        else:
            token='probe_'+uuid.uuid4().hex
            admission=await gate.try_probe(token) if arm=='D' else dict(accepted=True)
            if admission['accepted']:
                leases[token]=dict(arm=arm,begin=time.monotonic(),admission=admission)
                result=dict(accepted=True,token=token,before=before,admission=admission)
            else:result=dict(accepted=False,reason=admission['reason'],before=before)
        append(ROOT/'raw'/f'interference_probe_admission_{name}.jsonl',result)
        return result

    @app.post('/campaign/probe/release')
    async def release(request:Request):
        nonlocal failure
        body=await request.json();token=body['token'];assert token in leases
        lease=leases[token];failed=body.get('failed',False)
        assert type(failed) is bool
        if not failed:
            # A completed one-token response is the only success boundary.
            assert isinstance(body.get('output_token_ids'),list) and len(body['output_token_ids'])==1
            assert body.get('completion_tokens')==1
        if failed:failure=token
        if lease['arm']=='D':
            if not failed:await gate.first_token(token)
            await gate.leave(token,failed=failed)
        leases.pop(token)
        result=dict(released=True,failed=failed,arm=lease['arm'],begin=lease['begin'],end=time.monotonic(),
            task_completion=False,scope='INTERFERENCE_PROBE. Discarded one-token prefix sample.')
        append(ROOT/'raw'/f'interference_probe_admission_{name}.jsonl',dict(token=token,**result))
        return result

    @app.get('/campaign/probe/state')
    async def probe_state():return dict(time=time.monotonic(),leases=leases,failure=failure,
        admission=gate.state(),scope='Coordinator HTTP state only. No GPU-idle or future-deadline guarantee.')
    return app


def main():
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--port',type=int,default=18845)
    p.add_argument('--encoder',default='http://127.0.0.1:18840');p.add_argument('--prefill',default='http://127.0.0.1:18841')
    p.add_argument('--decoder',default='http://127.0.0.1:18842');args=p.parse_args()
    write(ROOT/'configs'/f'probe_proxy_{args.name}.json',dict(time=time.time(),args=vars(args),source_SHA256=digest(__file__),
        evidence='INTERFERENCE_PROBE coordinator',no_task_prefill_offload=True,
        policy='E probe starts only with zero E HTTP requests and P HTTP work. New vision may arrive during its non-preemptible kernels. '
            'D probe requires zero D streams and no pending admission. Native D waits until the actual probe sample completes.',
        unchanged_native_handler=True,no_future_information=True))
    uvicorn.run(application(args.name,args.encoder,args.prefill,args.decoder),host='127.0.0.1',port=args.port,log_level='warning')


if __name__=='__main__':main()
