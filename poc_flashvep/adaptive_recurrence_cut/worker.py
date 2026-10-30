"""One immutable fixed-cut experiment, on the released numerical execution path."""
import json
import os
import sys
import time
from pathlib import Path
from .common import ROOT,MODEL_PINS,VARIANTS,setup,write,append,load_config


def install():
    cfg=load_config()
    from ..looped_pd_abc import common
    common.ROOT=ROOT
    common.setup=setup
    common.append=append
    from ..looped_pd_b_numerical_diagnosis import common as numerical
    numerical.ROOT=ROOT
    os.environ['B_NUMERICAL_RUN']=os.environ['ARC_RUN']
    os.environ['B_NUMERICAL_VARIANT']=VARIANTS[cfg['model']]
    from ..looped_pd_b_numerical_diagnosis.patches import install as qualified
    qualified()
    from .staged import install as staging
    staging(cfg)
    from .state_probe import install as probe
    probe(cfg)
    import vllm_rlt.pd.engine as pd
    from .connector import CutConnector
    pd.PDStageConnector=CutConnector


if os.environ.get('ARC_RUN'):
    install()


def serve(engine,inputs,args,run,warmup=False,contract=None):
    from vllm_rlt import SamplingParams
    cfg=contract or load_config()
    offsets=[0.]*len(inputs) if warmup else cfg['arrival_offsets'][:len(inputs)]
    cuts=cfg.get('request_cuts',[cfg['cut']]*len(inputs))
    caps=cfg.get('request_caps',[args.cap]*len(inputs))
    start=time.monotonic()
    active,first,last,events,outputs,admitted={},{},{},{},{},{}
    index=0
    while index<len(inputs) or active:
        while index<len(inputs) and time.monotonic()>=start+offsets[index]:
            rid=f'r{run}_q{index}'
            seed=cfg['seed']+index
            if hasattr(engine,'_arc_cuts'):engine._arc_cuts[rid]=cuts[index]
            sampling=SamplingParams(max_tokens=caps[index],temperature=0.,seed=seed,
                                    min_loops=MODEL_PINS[args.model][2],max_loops=MODEL_PINS[args.model][2],
                                    exit_threshold=1.,ignore_eos=cfg.get('ignore_eos',True))
            engine.add_request(rid,inputs[index],sampling)
            admitted[rid]=time.monotonic()
            active[rid]=index
            events[rid]=[]
            index+=1
        for output in engine.step():
            now=time.monotonic();rid=output.request_id
            count=len(output.token_ids)
            previous=last.get(rid,(now,0))
            if count>previous[1]:
                first.setdefault(rid,now)
                events[rid].append(dict(time=now,previous_time=previous[0],new_tokens=count-previous[1],first=previous[1]==0))
                last[rid]=now,count
            if not output.finished:continue
            i=active.pop(rid)
            if any(depth!=MODEL_PINS[args.model][2] for depth in output.exit_depths):
                raise RuntimeError('Fixed recurrent depth changed')
            arrival=start+offsets[i]
            row=dict(model=args.model,cut=cuts[i],name=args.name,run=run,sample_id=i,
                     cohort_id=cfg.get('cohort_id','single'),repeat=cfg.get('repeat',0),
                     target=cfg.get('target_mask',[True]*len(inputs))[i],
                     prompt_id=cfg.get('prompt_ids',list(range(len(inputs))))[i],
                     input_ids=inputs[i],output_ids=list(output.token_ids),exit_depths=list(output.exit_depths),
                     input_tokens=len(inputs[i]),output_tokens=count,seed=cfg['seed']+i,
                     arrival=arrival,admission=admitted[rid],finish=now,
                     ttft_ms=1000*(first.get(rid,now)-arrival),
                     tpot_ms=1000*(now-first.get(rid,now))/max(1,count-1),
                     e2e_ms=1000*(now-arrival),token_events=events[rid],
                     finish_reason=str(output.finish_reason),warmup=warmup,gpus=cfg['gpus'],
                     chunk_tokens=cfg['chunk'],recurrent_depth=MODEL_PINS[args.model][2],
                     regime=cfg.get('regime','correctness'),co_batching=False,
                     prompt_source=cfg.get('prompt_source','Frozen ShareGPT token-prefix diagnostic'),
                     generation='Controlled ignore-EOS' if cfg.get('ignore_eos',True) else 'Natural EOS, bounded output',
                     evidence='MEASURED_HEAVY_CORRECTNESS' if cfg.get('verify_transfer') or cfg.get('state_probe') else 'MEASURED',
                     timing_scope='Scheduled arrival to final output. Includes delivery and queueing. No HTTP.')
            append(ROOT/'raw'/args.name/'requests.jsonl',row)
            outputs[i]=row
            print('DONE',args.name,cfg.get('cohort_id','single'),i,count,round(row['e2e_ms'],3),flush=True)
        if time.monotonic()-start>args.timeout:raise TimeoutError('Fixed-cut cohort watchdog')
        if not active and index<len(inputs):time.sleep(min(.001,max(0.,start+offsets[index]-time.monotonic())))
    if not warmup:write(ROOT/'raw'/args.name/f'window_{cfg.get("cohort_id","single")}.json',dict(start=start,end=time.monotonic(),scope='Finite mechanism cohort, not sustainable capacity'))
    return outputs


def install_metadata(engine,default_cut=None):
    """Commit one cut before P execution. No online cut selector is installed."""
    engine._arc_cuts={}
    if default_cut is None:default_cut=load_config()['cut']
    original=engine._send

    def send(destination,kind,**fields):
        if kind in ('reserve','prefill','commit'):
            tid=fields.get('tid')
            pending=engine.transfers.get(tid)
            if pending is not None:fields['arc_cut']=engine._arc_cuts.get(pending.request.request_id,default_cut)
        return original(destination,kind,**fields)

    engine._send=send


def install_local_endpoint(engine):
    """Use native D-stage reservation and Prefill. P keeps its resident model."""
    original_admit=engine._admit
    original_message=engine._message
    default_cut=load_config()['cut']

    def admit():
        for pending in engine.transfers.values():
            if pending.phase=='waiting' and engine._arc_cuts.get(pending.request.request_id,default_cut)==0:
                pending.arc_local=True
                pending.phase='local_waiting'
        original_admit()
        for pending in engine.transfers.values():
            if pending.phase!='local_waiting':continue
            tokens=len(pending.request.prompt_token_ids)+pending.request.sampling_params.max_tokens-1
            eligible=[p for p in engine.peers.values() if p.role=='decode'
                      and p.slots<p.client.info['transfer_slots']
                      and p.reserved_blocks+engine._blocks(p,tokens)<=p.client.info['num_blocks']-(p.client.info['watermark_blocks'] if p.slots else 0)]
            if not eligible:continue
            d=min(eligible,key=lambda p:p.reserved_blocks)
            pending.d=d.name;pending.p_released=True
            pending.d_reserved_blocks=engine._blocks(d,tokens)
            d.reserved_blocks+=pending.d_reserved_blocks;d.slots+=1
            pending.phase='local_reserving'
            pending.timings['admitted']=time.monotonic()-pending.created
            source=next(p for p in engine.peers.values() if p.role=='prefill')
            engine._send(d.name,'reserve',tid=pending.tid,tokens=pending.request.prompt_token_ids,
                         params=pending.request.sampling_params,trace_id=pending.trace_id,
                         peer=source.client.info['info']['agent'])

    def message(peer,record):
        pending=engine.transfers.get(record.get('tid'))
        if record['kind']=='reserved' and pending is not None and getattr(pending,'arc_local',False):
            if peer.name!=pending.d:raise RuntimeError('Wrong local-Prefill destination')
            pending.timings['d_reserved']=time.monotonic()-pending.created
            pending.phase='local_prefill'
            return
        return original_message(peer,record)

    engine._admit=admit
    engine._message=message


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text())
    install()
    from ..looped_pd_abc import worker as base
    base.ROOT=ROOT;base.setup=setup;base.append=append;base.serve=serve
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    base.prompts=lambda model,count,length:[item['input_ids'][model] for item in data[:count]]
    configuration=base.configuration

    def configure(model,width,chunk,graphs=False,asynchronous=False):
        from dataclasses import replace
        value=configuration(model,width,chunk,graphs,asynchronous)
        value['scheduler_config']=replace(value['scheduler_config'],max_num_batched_tokens=cfg.get('token_budget',chunk))
        value['execution_config']=replace(value['execution_config'],cuda_graph_max_batch_size=cfg.get('graph_rows',width))
        return value

    base.configuration=configure
    if cfg.get('cohorts'):
        def campaign(engine,inputs,args,run,warmup=False):
            if warmup:return {}
            final={}
            references={}
            for number,cohort in enumerate(cfg['cohorts']):
                contract=dict(cfg,**cohort)
                contract['cohort_id']=cohort['cohort_id']
                if hasattr(engine,'has_unfinished_requests'):
                    while engine.has_unfinished_requests():engine.step()
                serve(engine,inputs[:min(cfg['capacity'],len(inputs))],args,-100-number,warmup=True,contract=contract)
                while engine.has_unfinished_requests():engine.step()
                stats=lambda:{name:peer.client.stats() for name,peer in engine.peers.items()} if hasattr(engine,'peers') else engine.stats()
                before=stats()
                final=serve(engine,inputs,args,number,contract=contract)
                while engine.has_unfinished_requests():engine.step()
                write(ROOT/'raw'/cfg['name']/f'cohort_{cohort["cohort_id"]}_stats.json',
                      dict(before=before,after=stats(),contract=contract,
                           metrics=list(getattr(engine,'metrics',[]))))
                if cfg.get('enforce_output_parity'):
                    key=cohort['regime']
                    values={i:tuple(json.dumps(r[k],sort_keys=True) for k in ('input_ids','output_ids','exit_depths','finish_reason')) for i,r in final.items()}
                    if cohort['cut']==MODEL_PINS[cfg['model']][2]:references[key]=values
                    else:
                        expected=references.get(key)
                        if expected is None:raise RuntimeError('Missing prior full-PD parity anchor')
                        checks={i:v==expected.get(i) for i,v in values.items()}
                        write(ROOT/'raw'/cfg['name']/f'parity_{cohort["cohort_id"]}.json',checks)
                        if not all(checks.values()):raise RuntimeError(f'Unexplained output mismatch in {cohort["cohort_id"]}')
            return final
        base.serve=campaign
    import vllm_rlt.pd.engine as pd
    original=pd.PDEngine.__init__

    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        install_metadata(self)
        install_local_endpoint(self)

    pd.PDEngine.__init__=initialize
    write(ROOT/'raw'/cfg['name']/'arc_contract.json',cfg)
    flags=['worker','--model',cfg['model'],'--mode','pd',
           '--gpus',','.join(map(str,cfg['gpus'])),'--name',cfg['name'],
           '--requests',str(cfg['count']),'--length','0','--cap',str(cfg['cap']),
           '--chunk',str(cfg['chunk']),'--concurrency',str(cfg['capacity']),
           '--prefill','1','--timeout',str(cfg.get('timeout',600)),'--wavefront']
    if cfg.get('graphs',True):flags.append('--graphs')
    if cfg.get('warmup',True):flags.append('--warmup')
    if cfg['model']=='ouro' and cfg.get('asynchronous',True):flags.append('--asynchronous')
    if cfg.get('profile'):flags.append('--profile')
    os.environ['ABC_ALLOW_BOUNDARY_PD']='1'
    sys.argv=flags
    base.main()


if __name__=='__main__':main()
