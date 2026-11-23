"""Separate coordinator request credit from proven source-buffer retirement."""
import atexit
import time


def safe_completion(kv_retired,fence_complete,hidden_required,hidden_acked,hidden_owner_released):
    return bool(kv_retired and fence_complete and (not hidden_required or (hidden_acked and hidden_owner_released)))


def release_owned(owned,rid):
    if rid not in owned:return False
    owned.remove(rid);return True


class Retirement:
    def __init__(self,cfg,role,folder,adapter,channel):
        self.cfg,self.role,self.folder,self.adapter,self.channel=cfg,role,folder,adapter,channel
        self.enabled=role=='prefill' and bool(cfg.get('source_credit_release_on_retirement') or cfg.get('P_critical_trace'))
        self.pending={};self.sent=set();self.rows=[];self.closed=False
        atexit.register(self.close)

    def before_ack(self,scheduler,value):
        if not self.enabled:return {}
        result={}
        for rid in value.finished_sending or ():
            if not rid.startswith('q'):continue
            request=scheduler.requests[rid]
            deferred=bool(scheduler.defer_block_free and request.last_sched_seq>scheduler.processed_step_seq)
            result[rid]=dict(native_ack_before=time.monotonic(),
                required_fence=int(scheduler.sched_step_seq) if deferred else 0,
                native_deferred_free=deferred,last_request_sched_seq=int(request.last_sched_seq))
        return result

    def after_ack(self,scheduler,before):
        if not self.enabled:return
        for rid,row in before.items():
            assert rid not in scheduler.requests,'Wait until the unchanged native retirement call succeeds'
            assert rid not in self.pending and rid not in self.sent
            self.pending[rid]=dict(row,native_ack_after=time.monotonic(),native_request_removed=True)

    def tick(self,scheduler):
        if not self.enabled or not self.pending:return
        transport=self.adapter.transport if self.adapter is not None else None
        if transport is not None:transport.reap()
        for rid,value in list(self.pending.items()):
            required=transport is not None
            ack=transport.directory/(rid+'.ACK.json') if required else None
            acked=not required or ack.exists()
            released=not required or (rid not in transport.parts and rid not in transport.owners)
            fenced=int(scheduler.processed_step_seq)>=value['required_fence']
            if not safe_completion(value['native_request_removed'],fenced,required,acked,released):continue
            now=time.monotonic();external=rid.split('-')[0]
            row=dict(value,event='SOURCE_RESOURCE_RETIREMENT_OBSERVED',request_id=external,source_request_id=rid,
                hidden_required=required,hidden_ACKed=acked,hidden_owner_released=released,
                native_fence_complete=fenced,processed_step_seq=int(scheduler.processed_step_seq),time=now)
            self.rows.append(row);self.sent.add(rid);del self.pending[rid]
            if self.cfg.get('source_credit_release_on_retirement') and self.cfg.get('continuous_source_credit_limit',0):
                self.channel.send(dict(kind='source_resources_retired',request_id=external,time=now,
                    native_request_removed=True,hidden_ACKed=acked,hidden_owner_released=released,native_fence_complete=fenced))

    def close(self):
        if self.closed:return
        self.closed=True
        from poc_flashvep.dense_ar_longbench400_pd.common import write
        write(self.folder/f'source_resource_retirement_{self.role}.json',dict(rows=self.rows,pending=list(self.pending),
            enabled=self.enabled,policy_enabled=bool(self.cfg.get('source_credit_release_on_retirement')),
            scope='Never free a native KV block manually. Native ACK processing and its GPU fence remain authoritative. '
                'Hidden source reuse follows the original completed-READ ACK validation and reap. '
                'Only the independent coordinator request-window credit can return earlier. '
                'First client-token release remains a fallback. No new D active-decode admission is introduced.'))
