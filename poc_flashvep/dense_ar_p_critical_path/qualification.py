"""Test the changed call and retirement seams before performance dispatch."""
import argparse
import fcntl
import json
import subprocess
import time
from .common import ROOT,PRIOR,REPO,PYTHON,write,digest,copy_data,bind,preserve,resources
from .bootstrap import baseline
from .capsule import build


def prepare(labels=('Layer','Hybrid'),lanes=(0,1),plan_name='retirement_qualification_r2.json'):
    runner=bind();items=[]
    from ..dense_ar_output_aware_v2.work_contract import regenerate
    for lane,label in zip(lanes,labels):
        pair=[2*lane,2*lane+1]
        base=baseline(label,f'PC_Metadata_{label}',pair)
        source=build(json.loads((PRIOR/'configs'/f'V2_C3_Heldout192Replay_q2.0_s0_{label}.json').read_text())['source_capsule'])
        strict=[]
        for version in ('Original','ObservedRetirement'):
            cfg=dict(base);name=f'PC_Q_{label}_{version}_Strict8_ArrivalFix2'
            cfg.update(name=name,dataset='compiler_control8_original_v1.json',count=8,
                dataset_SHA256=digest(copy_data('compiler_control8_original_v1.json')),
                output_cap=16,ignore_eos=True,serial=True,controlled_pipeline=True,
                continuous_source_credit_limit=0,correctness_only=True,axis_state_capture=True,
                axis_capture_hidden_last_only=False,axis_import_hash_after_model=False,
                gpu_state_digest=False,verify_GPU_digest_with_CPU=False)
            for key in ('replay_output_lengths','diagnostic_controlled_output_replay','qualification_gate_v2'):
                cfg.pop(key,None)
            cfg['arrival_offsets']=[0.]*8
            if version!='Original':
                cfg.update(source_capsule=source.name,source_capsule_SHA256=digest(source),
                    P_event_timing=True,P_critical_trace=True,source_credit_release_on_retirement=True)
            regenerate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']])
            runner.save(cfg);strict.append(name)
        policies=[]
        for enabled in ((False,) if label=='StockToken' else (False,True)):
            label2='Retirement' if enabled else 'Control';jobs=[]
            for count,dataset in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
                cfg=dict(base);name=f'PC_Q_{label}_{label2}_Natural{count}'
                cfg.update(name=name,dataset=dataset,dataset_SHA256=digest(copy_data(dataset)),count=count,
                    source_capsule=source.name,source_capsule_SHA256=digest(source),
                    P_event_timing=False,P_critical_trace=True,source_credit_release_on_retirement=enabled,
                    correctness_only=True,axis_state_capture=True,axis_capture_hidden_last_only=True,
                    axis_import_hash_after_model=True,gpu_state_digest=True,verify_GPU_digest_with_CPU=False,
                    ignore_eos=False,qps=3.2,seed=2026112308)
                import numpy as np
                cfg['arrival_offsets']=np.r_[0.,np.cumsum(np.random.default_rng(cfg['seed']).exponential(1/cfg['qps'],count-1))].tolist()
                for key in ('replay_output_lengths','diagnostic_controlled_output_replay','qualification_gate_v2'):
                    cfg.pop(key,None)
                if label=='StockToken':cfg.pop('source_credit_release_on_retirement')
                regenerate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']])
                runner.save(cfg);jobs.append(name)
            policies.append(dict(enabled=enabled,label=label2,jobs=jobs,
                gate=f'pc_{label.lower()}_{label2.lower()}_composition.json',signature=runner.signature(cfg)))
        items.append(dict(label=label,lane=lane,pair=pair,strict=strict,policies=policies,
            source_capsule=source.name,source_capsule_SHA256=digest(source)))
    path=ROOT/'configs'/plan_name;assert not path.exists()
    write(path,dict(time=time.time(),items=items,
        scope='Original and changed source use identical serial execution geometry for exact differential state tests. '
            'The observer adds events only. Natural32/400 test both false and true credit policies on the same changed source. '
            'Native KV retirement and hidden ownership remain authoritative. These captures are correctness-only.'))
    children=[]
    for item in items:
        log=ROOT/'logs'/f'{path.stem}_lane{item["lane"]}.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.qualification',
                'run','--lane',str(item['lane']),'--plan',plan_name],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(lane=item['lane'],pid=child.pid,log=str(log.relative_to(ROOT))))
    write(ROOT/'state'/(path.stem+'_controller.json'),dict(time=time.time(),children=children))
    return children


def differential(original,changed):
    import torch
    from ..dense_ar_longbench400_pd.qualification_gate import KV_equal
    torch.set_num_threads(1)
    paths=[ROOT/'raw'/name for name in (original,changed)]
    states=[json.loads((p/'axis_events_decode.json').read_text())['states'] for p in paths]
    outputs=[{r['sample_id']:r for r in map(json.loads,(p/'requests.jsonl').read_text().splitlines())} for p in paths]
    assert set(outputs[0])==set(outputs[1]);rows=[]
    for index in sorted(outputs[0]):
        rid='q'+str(index);a,b=[state[rid] for state in states]
        hidden=[torch.load(state['tensor_file'],map_location='cpu',weights_only=False)['hidden'] for state in (a,b)]
        rows.append(dict(request_id=rid,full_prompt_KV_exact=KV_equal(a['KV'],b['KV']),
            captured_hidden_exact=torch.equal(*hidden),greedy_exact=outputs[0][index]['output_ids']==outputs[1][index]['output_ids']))
    observer=json.loads((paths[1]/'P_model_forward_events_prefill.json').read_text())
    assert observer['rows'],'The actual event seam must execute'
    passed=all(all(row[k] for k in ('full_prompt_KV_exact','captured_hidden_exact','greedy_exact')) for row in rows)
    result=dict(time=time.time(),passed=passed,original=original,changed=changed,rows=rows,
        actual_observed_model_calls=len(observer['rows']),evidence='CORRECTNESS_ONLY',
        scope='Exact differential test against the qualified original same-partition same-geometry runtime. '
            'Historical full-model teacher proof remains separate. No new cross-geometry numerical equality is assumed.')
    write(ROOT/'analysis'/(changed+'_differential.json'),result)
    assert passed,result
    return result


def run(lane,plan_name='retirement_qualification_r2.json'):
    runner=bind()
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from .quality import compare as quality
    item=next(v for v in json.loads((ROOT/'configs'/plan_name).read_text())['items'] if v['lane']==lane)
    pair=item['pair'];owner=resources();progress=[]
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);owner.stop(pair)
        try:
            for name in item['strict']:
                owner.stop(pair);assert runner.execute(name),name
                result=audit(name);assert result['passed'] and result['decode_full36_observation_complete']
            exact=differential(*item['strict'])
            for policy in item['policies']:
                stages=[];proof=None
                try:
                    for name in policy['jobs']:
                        owner.stop(pair);assert runner.execute(name),name;summarize(name)
                        state=audit(name);assert state['passed'] and state['decode_full36_observation_complete']
                        assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
                        score=quality(name);assert score['passed'],score
                        folder=ROOT/'raw'/name
                        events=[json.loads(v) for v in (folder/'credit_events.jsonl').read_text().splitlines()]
                        released=[v for v in events if v['event']=='RELEASE_SAFE_SOURCE_RETIREMENT']
                        retired=json.loads((folder/'source_resource_retirement_prefill.json').read_text())
                        assert retired['enabled'] and not retired['pending'],retired
                        if policy['enabled']:assert released,'Exercise the actual early credit path'
                        else:assert not released
                        assert all(v['native_request_removed'] and v['native_fence_complete'] and
                            v['hidden_ACKed'] and v['hidden_owner_released'] for v in retired['rows'])
                        stages.append(dict(name=name,state=state,quality=score,retired=len(retired['rows']),
                            actual_early_credit_releases=len(released)))
                    proof=dict(time=time.time(),passed=True,full400_passed=True,quality_passed=True,strict=exact,
                        stages=stages,signature=policy['signature'],source_capsule_SHA256=item['source_capsule_SHA256'],
                        evidence='CORRECTNESS_ONLY')
                except Exception as error:
                    proof=dict(time=time.time(),passed=False,error=repr(error),evidence='FAILED')
                write(ROOT/'state'/policy['gate'],proof);progress.append(dict(label=policy['label'],passed=proof['passed'],error=proof.get('error')))
                write(ROOT/'state'/f'retirement_qualification_lane{lane}_progress.json',dict(time=time.time(),results=progress))
                print('PC_RETIREMENT_GATE',item['label'],progress[-1],flush=True)
                if not proof['passed']:break
        except Exception as error:
            write(ROOT/'state'/f'{__import__("pathlib").Path(plan_name).stem}_lane{lane}_failure.json',dict(time=time.time(),error=repr(error),evidence='FAILED'))
            raise
        finally:owner.restore(pair);preserve()
    write(ROOT/'state'/f'retirement_qualification_lane{lane}_complete.json',dict(time=time.time(),results=progress,
        complete=len(progress)==len(item['policies']) and all(v['passed'] for v in progress)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','stock','run'));p.add_argument('--lane',type=int)
    p.add_argument('--plan',default='retirement_qualification_r2.json')
    args=p.parse_args()
    print(prepare() if args.action=='prepare' else prepare(('StockToken',),(3,),'stock_observer_qualification_r2.json')
        if args.action=='stock' else run(args.lane,args.plan),flush=True)
