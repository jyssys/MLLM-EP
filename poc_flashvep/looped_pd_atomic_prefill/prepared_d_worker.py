"""Reuse the existing native prepared-KV barrier for D-readiness diagnostics."""
from .common import *


def main():
    from . import rlt_worker
    original=rlt_worker.serve
    def serve(engine,inputs,args,run,warmup=False,contract=None):
        cfg=contract or config();folder=ROOT/'raw'/cfg['name']
        if warmup:return original(engine,inputs,args,run,True,contract)
        write(folder/'prepared_D_marker.json',dict(count=cfg['count'],time=time.monotonic()))
        send=engine._send;held=[]
        def hold(destination,kind,**fields):
            if kind!='commit':return send(destination,kind,**fields)
            held.append((destination,fields))
            if len(held)==cfg['count']:
                for target,payload in held:send(target,'commit',**payload)
        engine._send=hold
        try:result=original(engine,inputs,args,run,False,contract)
        finally:engine._send=send
        ready=list(folder.glob('prepared_D_ready_*.json'));assert len(ready)==1,'Missing unique native valid-KV D barrier'
        boundary=json.loads(ready[0].read_text());assert boundary['valid_KV_and_hidden'] and boundary['count']==cfg['count']
        release=boundary['time'];offsets=cfg['prepared_D_arrival_offsets']
        events={r['index']:r for r in [json.loads(line) for line in (folder/'prepared_D_release_events.jsonl').read_text().splitlines()]}
        for i,r in result.items():
            arrival=release+offsets[i]
            assert r['first_token']>=arrival,'Decode completed before its scheduled ready time'
            value=dict(r,preparation_arrival=r['arrival'],preparation_admission=r['admission'],
                arrival=arrival,admission=events[i]['actual'],
                ttft_ms=1000*(r['first_token']-arrival),e2e_ms=1000*(r['finish']-arrival),
                evidence='MEASURED_NATIVE_PREPARED_STAGGERED_D_ONLY',
                scope='Native NIXL import and valid-state synchronization precede this D-only timer. P preparation remains in original requests.jsonl, not in this diagnostic latency.')
            append(folder/'prepared_D_requests.jsonl',value)
        write(folder/'prepared_D_window.json',dict(start=release,end=max(r['finish'] for r in result.values()),
            last_arrival=release+offsets[-1],ready_pattern=cfg.get('D_ready_pattern','CONTROLLED_READY_TIMES'),
            scope='Finite prepared-KV cohort. Not sustainable online capacity. No Dynamic-A execution.'))
        return result
    rlt_worker.serve=serve;rlt_worker.main()


if __name__=='__main__':main()
