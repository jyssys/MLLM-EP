"""Observable-state decisions. No future arrival or output-length inputs."""
import time
from .common import ROOT,append

def token_candidates(n,chunk,depth,cuts):
    """Match loop fractions before legal token-chunk alignment and deduplication."""
    return sorted({0,n}|{min(n,max(min(chunk,n),int(n*c/depth)//chunk*chunk)) for c in cuts if c>0})

def install(engine,cfg):
    depth=cfg['depth'];engine._arc_cuts={};engine._ltp_prefixes={}
    last_output={};decisions={};original_send=engine._send
    def state():
        now=time.monotonic()
        active=[w for w in engine.transfers.values() if w.phase=='decode' and not w.finished and w.request.generated_token_ids]
        ages=[now-last_output.get(w.tid,now) for w in active]
        p=[w for w in engine.transfers.values() if w.phase in ('waiting','reserving','prefill')]
        d=next(v for v in engine.peers.values() if v.role=='decode')
        return dict(active_decode=len(active),p_backlog=len(p),
                    d_continuations=sum(w.phase=='decode' and not w.finished and not w.request.generated_token_ids for w in engine.transfers.values()),
                    max_decode_age_s=max(ages,default=0.),
                    d_free_blocks=d.client.info['num_blocks']-d.reserved_blocks,
                    tpot_slo_s=cfg.get('tpot_slo_ms',1000.)/1000.)
    def choose(rid,tokens):
        s=state();n=len(tokens);chunk=cfg['chunk']
        if cfg.get('dynamic'):
            # Conservative model from disjoint calibration. It is not an SLO guarantee.
            unit=cfg.get('prefill_ms_per_token',.5)*n/1000.
            slack=s['tpot_slo_s']-s['max_decode_age_s']
            spare=max(0.,slack)*cfg.get('slack_fraction',.5)
            pressure=s['p_backlog']
            fraction=(min(.75,spare/max(unit,.00001)) if pressure>=1 else 0.)
            if s['active_decode']==0 and pressure>=1:fraction=max(fraction,.5)
            if cfg['axis']=='loop':
                candidates=cfg.get('cut_candidates',[0,depth//2,depth])
                cut=min((c for c in candidates if (depth-c)/depth<=fraction+1e-9),default=depth)
                prefix=n
            else:
                candidates=token_candidates(n,chunk,depth,cfg.get('cut_candidates',[0,depth//4,depth//2,3*depth//4,depth]))
                prefix=min((k for k in candidates if (n-k)/n<=fraction+1e-9),default=n);cut=depth
        else:
            cut=cfg.get('cut',depth)
            prefix=(min(n,int(n*cfg.get('prefix_fraction',1.))//chunk*chunk) if cfg['axis']=='token' else n)
            if cfg['axis']=='token' and cfg.get('prefix_fraction',1.)>0:prefix=max(min(chunk,n),prefix)
        if prefix==0:cut=0
        forced=getattr(engine,'ltp_forced_plan',None)
        if forced is not None:
            cut,fraction=forced
            if cfg['axis']=='token':
                prefix=min(n,int(n*fraction)//chunk*chunk)
                if fraction>0:prefix=max(min(chunk,n),prefix)
                if not prefix:cut=0
        engine._arc_cuts[rid]=cut;engine._ltp_prefixes[rid]=prefix
        row=dict(request_id=rid,time=time.monotonic(),event='INITIAL_PLAN',axis=cfg['axis'],
                 cut=cut,prefix_k=prefix,prompt_tokens=n,state=s,online=cfg.get('dynamic',False))
        decisions[rid]=row;append(ROOT/'raw'/cfg['name']/'online_decisions.jsonl',row)
    engine.ltp_choose=choose
    engine.ltp_update_contract=lambda values:cfg.update(values)
    def send(destination,kind,**fields):
        if kind in ('reserve','prefill','commit'):
            w=engine.transfers.get(fields.get('tid'))
            if w is not None:
                rid=w.request.request_id
                fields['arc_cut']=engine._arc_cuts.get(rid,cfg.get('cut',depth))
                fields['prefix_k']=engine._ltp_prefixes.get(rid,len(w.request.prompt_token_ids))
        return original_send(destination,kind,**fields)
    engine._send=send
    message=engine._message
    def receive(peer,m):
        w=engine.transfers.get(m.get('tid') or m.get('request_id'))
        if m['kind']=='frontier_ready':
            if w is None:raise RuntimeError('Stale frontier request')
            s=state();n=len(w.request.prompt_token_ids)
            remain=(depth-m['arc_cut'])/depth if cfg['axis']=='loop' else (n-m['prefix_k'])/n
            cost=cfg.get('prefill_ms_per_token',.5)*n*remain/1000.
            slack=s['tpot_slo_s']-s['max_decode_age_s']
            accept=s['active_decode']==0 or cost<=max(0.,slack)*cfg.get('slack_fraction',.5)
            reason='D_SLACK_CREDIT_ACCEPT' if accept else 'D_SLACK_RECHECK_REFUSED'
            if not accept:
                engine._arc_cuts[w.request.request_id]=depth
                engine._ltp_prefixes[w.request.request_id]=n
            append(ROOT/'raw'/cfg['name']/'online_decisions.jsonl',dict(
                request_id=w.request.request_id,time=time.monotonic(),event='FRONTIER_RECHECK',
                accept=accept,reason=reason,predicted_continuation_s=cost,state=s,
                cut=m['arc_cut'],prefix_k=m['prefix_k']))
            engine._send(w.p,'frontier_decision',tid=w.tid,accept=accept,reason=reason)
            return
        if w and m['kind']=='output':last_output[w.tid]=time.monotonic()
        if m['kind']=='commit' and w:
            engine._arc_cuts[w.request.request_id]=m.get('arc_cut',depth)
            engine._ltp_prefixes[w.request.request_id]=m.get('prefix_k',len(w.request.prompt_token_ids))
        return message(peer,m)
    engine._message=receive
