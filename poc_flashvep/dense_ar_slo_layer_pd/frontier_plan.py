"""Simulate only observed requests and legal non-preemptive cohort boundaries."""


def simulate(now,source_ms,source_frame_ms,slots,maximum_destination,caps,
             available,cost,wait_whole_source=False):
    assert caps and slots>0 and maximum_destination>0
    frames=[list(range(i,min(i+slots,len(caps)))) for i in range(0,len(caps),slots)]
    weights=[source_frame_ms(len(f)) for f in frames];total=sum(weights)
    assert total>0
    readiness={};at=now
    for ids,weight in zip(frames,weights):
        at+=source_ms*(weight/total)/1000
        for index in ids:readiness[index]=at
    if wait_whole_source:readiness={i:at for i in readiness}
    waiting=list(range(len(caps)));clock=available;cohorts=[];first={}
    while waiting:
        clock=max(clock,readiness[waiting[0]])
        ready=[i for i in waiting if readiness[i]<=clock+1e-9][:maximum_destination]
        assert ready
        handoff,suffix,commit,decode_step=cost(len(ready))
        begin=clock+handoff/1000;output=begin+(suffix+commit)/1000
        decode=decode_step*max(0,max(caps[i] for i in ready)-1)
        for index in ready:first[index]=output
        cohorts.append(dict(indices=ready,batch=len(ready),handoff_start=clock,
            suffix_start=begin,first_output=output,retire=output+decode/1000,
            handoff_ms=handoff,suffix_ms=suffix,commit_ms=commit,decode_ms=decode,
            decode_step_bound_ms=decode_step))
        clock=cohorts[-1]['retire'];waiting=[i for i in waiting if i not in ready]
    return dict(source_frontiers=readiness,source_ready=max(readiness.values()),
        first_outputs=first,first_suffix_start=cohorts[0]['suffix_start'],
        group_first_output=max(first.values()),group_retire=clock,cohorts=cohorts,
        service_ms=sum(c['handoff_ms']+c['suffix_ms']+c['commit_ms']+c['decode_ms'] for c in cohorts),
        maximum_decode_step_bound_ms=max(c['decode_step_bound_ms'] for c in cohorts),
        scope='A deterministic empirical plan for already-observed source requests. It does not observe future arrivals or certify a physical idle window.')


def for_candidate(policy,state,layers,negotiation,available):
    caps=list(state.group_output_caps) if state.group_output_caps else [state.maximum_group_output_tokens]*state.group_size
    slots=min(policy.cfg['P_slots'],policy.cfg['token_budget']//(policy.current_length or policy.cfg['length']))
    maximum=min(policy.cfg.get('D_admission_batch',16),policy.cfg.get('D_token_budget',128000)//(policy.current_length or policy.cfg['length']))
    def cost(batch):
        return (policy._cost('handoff_ms',layers,batch),policy._cost('D_suffix_ms',layers,batch),
            policy._cost('D_commit_ms',layers,batch) if 'D_commit_ms' in policy.costs else 0.,policy.decode_bound(batch))
    return simulate(state.now+negotiation/1000,policy._cost('P_prefix_ms',layers,state.group_size),
        lambda n:policy._cost('P_prefix_ms',layers,n),slots,maximum,caps,available,cost,
        policy.cfg.get('D_wait_for_source_macro',False))
