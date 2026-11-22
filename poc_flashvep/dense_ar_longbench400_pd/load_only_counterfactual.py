"""Test offered-load causality with fixed requests,output,cut,source and batch."""
import argparse
import json
import time
from .common import ROOT,write
from .run import save
from .confirmation_analysis import load,ratio
from .service_accounting import compare


PLAN='load_only_I0_O8_Cut2_frozen_v1.json'


def plan():
    source=json.loads((ROOT/'configs/heldout64_confirmation_protocol_r2.json').read_text())
    low=[r for r in source['cells'] if (r['input_bucket'],r['output'],r['cut'])==(0,8,2)]
    assert len(low)==3
    qps=8.29212396320837
    jobs=[]
    cells=[]
    for index,cell in enumerate(low):
        pair={}
        for axis in ('Native','Layer','Token'):
            cfg=json.loads((ROOT/'configs'/(cell[axis]+'.json')).read_text())
            assert cfg['gpus']==[2,3] and cfg['output_cap']==8 and cfg['controlled_pipeline']
            assert cfg['token_budget']==8192 and cfg['source_credit_limit']==4
            name=f'LoadOnly_I0_O8_C2_s{index}_{axis}_v1'
            old_q=cfg['qps']
            cfg.update(name=name,qps=qps,arrival_offsets=[t*old_q/qps for t in cfg['arrival_offsets']],
                load_only_control=cell[axis],
                scope='Same source capsule,pair2/3,request IDs,input,output8,cut2,credit4 and actual1/1/1 as the original low-load control. '
                    'Change only offered QPS and scale the same normalized Poisson interarrivals. '
                    'Three predeclared start blocks. Retain either ordering or a tie. No repeat-until-crossover rule.')
            save(cfg)
            jobs.append(name)
            pair[axis]=dict(low=cell[axis],high=name)
        cells.append(dict(start=index,seed=cell['seed'],low_QPS=old_q,high_QPS=qps,jobs=pair))
    path=ROOT/'configs'/PLAN
    assert not path.exists()
    value=dict(jobs=jobs,cells=cells,source_capsule_SHA256=source['source_capsule_SHA256'],
        evidence='FROZEN_SINGLE_VARIABLE_LOAD_COUNTERFACTUAL',
        hypotheses=['At low load,the shorter combined source/destination service can favor Layer.',
            'Near source capacity,Token source-service saving can amplify in its source queue.',
            'The predicted reversal can fail or remain uncertain;keep that outcome.'],
        scope='High load was chosen from the existing proper-pipeline-rate discovery,not held-out outcomes. '
            'Reuse all three earlier low-load controls. Do not rerun them until a desired sign appears. '
            'No output,input,partition,source,batch,GPU-pair,SLO or credit change. '
            'This is a C0 causal mechanism test,not C3 continuous-serving superiority.')
    write(path,value)
    write(ROOT/'jobs'/PLAN,dict(jobs=jobs,stop_on_failure=True))
    print('LOAD_ONLY_COUNTERFACTUAL_PLANNED',len(jobs),flush=True)


def analyze():
    contract=json.loads((ROOT/'configs'/PLAN).read_text())
    values={load:{a:[] for a in ('Native','Layer','Token')} for load in ('low','high')}
    stages=[]
    for cell in contract['cells']:
        for axis,jobs in cell['jobs'].items():
            configs={}
            for label,name in jobs.items():
                m,cfg=load(name,contract['source_capsule_SHA256'])
                if m is None:
                    print('LOAD_ONLY_COUNTERFACTUAL_PENDING',name,flush=True)
                    return
                values[label][axis].append(m)
                configs[label]=cfg
            for field in ('dataset_SHA256','seed','gpus','output_cap','token_budget','max_num_seqs',
                    'source_credit_limit','axis_cut','execution_family'):
                assert configs['low'].get(field)==configs['high'].get(field),(axis,field)
            assert all(abs(a*configs['low']['qps']/configs['high']['qps']-b)<1e-12
                for a,b in zip(configs['low']['arrival_offsets'],configs['high']['arrival_offsets']))
        stages.append(dict(start=cell['start'],low=compare(cell['jobs']['Layer']['low'],cell['jobs']['Token']['low']),
            high=compare(cell['jobs']['Layer']['high'],cell['jobs']['Token']['high'])))
    results={}
    for label,v in values.items():
        tests={f:ratio([r[f] for r in v['Layer']],[r[f] for r in v['Token']]) for f in
            ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')}
        safety={a:dict(native_relative_5percent=all(ratio([r[f] for r in v[a]],
                [r[f] for r in v['Native']])['CI95'][1]<=1.05 for f in ('ITL_p99_ms','TPOT_p95_ms')),
            absolute_decode=all(r['decode_SLO_attainment']>=.95 for r in v[a])) for a in ('Layer','Token')}
        results[label]=dict(Layer_over_Token=tests,safety=safety)
    reversal=(results['low']['Layer_over_Token']['TTFT_p95_ms']['CI95'][1]<1 and
        results['high']['Layer_over_Token']['TTFT_p95_ms']['CI95'][0]>1 and
        all(results['low']['safety']['Layer'].values()) and all(results['high']['safety']['Token'].values()))
    write(ROOT/'analysis/load_only_I0_O8_Cut2_counterfactual_v1.json',dict(time=time.time(),results=results,
        start_host_stage_differences=stages,load_only_reversal_confirmed=reversal,
        evidence='MEASURED_ONLINE_WITH_TRACE_ANALYSIS',
        scope='Three paired independent start blocks. Exact same source,pair,requests,partition,output8 and actual1/1/1. '
            'Only offered load changes. Pointwise paired empirical95percent CIs,27 resamples. '
            'A reversal does not imply Layer beats Native or best-tuned C3 Token. '
            'Mean host boundaries locate queue effects,not exclusive GPU operators. Finite16-request traces do not establish sustainable capacity.'))
    print('LOAD_ONLY_COUNTERFACTUAL_REVERSAL',reversal,results,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    plan() if args.plan else analyze()


if __name__=='__main__':
    main()
