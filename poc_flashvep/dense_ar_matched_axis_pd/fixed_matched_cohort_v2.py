"""Track actual local suffix submissions, not imported-KV progress hints."""
import hashlib
import inspect
import subprocess
import sys
from . import fixed_matched_cohort as base


def external(rid):
    return rid.split('-')[0]


class SubmissionFrontier:
    def __init__(self, cfg, channel):
        self.cfg=cfg
        self.channel=channel
        self.submitted=set()

    def group(self,rid):
        rid=external(rid)
        if not (rid.startswith('q') and rid[1:].isdigit()):
            return None
        channel=self.channel()
        assert channel is not None and rid in channel.by_request, ('Missing actual cohort registration',rid)
        return channel.groups[channel.by_request[rid]]

    def has_selected(self,scheduler,selected):
        return any(count>0 and self.group(rid) is not None and external(rid) not in self.submitted
                   for rid,count in selected.items())

    def defer_running(self,scheduler,request,selected):
        group=self.group(request.request_id)
        if group is None:
            return False
        rid=external(request.request_id)
        if rid not in self.submitted:
            return self.has_selected(scheduler,selected)
        return not group['members']<=self.submitted

    def after_schedule(self,result):
        for internal,count in result.num_scheduled_tokens.items():
            group=self.group(internal)
            rid=external(internal)
            if group is None or rid in self.submitted:
                continue
            index=int(rid[1:])
            expected=(self.cfg['token_suffix_by_request'][index]
                      if self.cfg.get('token_axis') else self.cfg['length'])
            assert count==expected, ('A local suffix must be a complete, first preparation query',rid,count,expected)
            self.submitted.add(rid)


def install(cfg,role):
    if role!='decode':
        return
    from .common import ROOT,write
    from ..dense_ar_slo_layer_pd import worker as qualified
    original_client=qualified.Client
    connection={}
    path=ROOT/'raw'/cfg['name']/'matched_first_token_barrier.jsonl'
    def record(value):
        import json
        with path.open('a') as stream:
            stream.write(json.dumps(value)+'\n')
    def client(*a,**kw):
        channel=base.FirstTokenBarrier(original_client(*a,**kw),record)
        connection['channel']=channel
        return channel
    qualified.Client=client
    if cfg['evaluation_arm']=='Native':
        return
    from vllm.v1.core.sched import scheduler as module
    frontier=SubmissionFrontier(cfg,lambda:connection.get('channel'))
    original=inspect.getsource(module.Scheduler.schedule)
    code=base.transform_schedule(original)
    scope=dict(module.__dict__,_matched_defer_running=frontier.defer_running,
               _matched_has_selected_prefill=frontier.has_selected)
    exec(compile(code,__file__+':local_submission_frontier','exec'),scope)
    scheduled=scope['schedule']
    def guarded(self,*a,**kw):
        result=scheduled(self,*a,**kw)
        frontier.after_schedule(result)
        return result
    module.Scheduler.schedule=guarded
    write(ROOT/'raw'/cfg['name']/'matched_cohort_scheduler_source.json',dict(
        original_SHA256=hashlib.sha256(original.encode()).hexdigest(),
        executed_SHA256=hashlib.sha256(code.encode()).hexdigest(),source=code,
        readiness_contract='Actual local full-suffix scheduled-token receipts, not num_computed_tokens from remote KV.',
        scope='Register cohort members from actual add commands. A running Decode request waits until every member has a local suffix submission. Query-count assertions reject unexpected replay or partial preparation. Model forward and KV management are unchanged.'))


def coordinator_main():
    previous=subprocess.Popen
    class Popen(previous):
        def __init__(self,args,*a,**kw):
            if isinstance(args,(list,tuple)) and '-m' in args:
                args=list(args);index=args.index('-m')+1
                if args[index]=='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort':
                    args[index]='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort_v2'
            super().__init__(args,*a,**kw)
    subprocess.Popen=Popen
    return base.coordinator_main()


def main():
    if '--role' not in sys.argv:
        return coordinator_main()
    from ..dense_ar_slo_layer_pd import native
    previous=native.stage
    def stage(cfg,role,address):
        install(cfg,role)
        return previous(cfg,role,address)
    native.stage=stage
    native.main()


if __name__=='__main__':
    main()
