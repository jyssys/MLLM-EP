"""Select the small rule on historical discovery, before new confirmation."""
import collections
import fcntl
import json
import statistics
import time
from .common import ROOT,PREVIOUS,QUIET_LOCK,write,digest,preserve
from .axis_oracle import predicted,pressure_rule
from .replay import Cost


def main():
    preserve()
    source=ROOT/'analysis/historical_replay_validation.json'
    document=json.loads(source.read_text())
    lookup={(r['method'],r['layers'],r['output'],r['batch']):Cost(**r['cost_seconds'])
        for r in document['models']}
    starts=json.loads((PREVIOUS/'analysis/independent_start_statistics.json').read_text())['starts']
    groups=collections.defaultdict(dict)
    for r in starts:
        if r['stage'] not in ('batch1_discovery','cohort2_discovery') or r['method'] not in ('Layer','Token'):
            continue
        groups[r['stage'],r['layers'],r['cap'],r['QPS'],r['seed'],r['count'],r['admission']][r['method']]=r
    cells=[]
    for key,values in sorted(groups.items()):
        if set(values)!=set(('Layer','Token')):
            continue
        stage,m,o,q,seed,count,b=key
        if count%18 or any((a,m,o,b) not in lookup for a in ('Layer','Token')):
            continue
        cfgs={a:json.loads((PREVIOUS/'configs'/(r['name']+'.json')).read_text()) for a,r in values.items()}
        assert cfgs['Layer']['arrival_offsets']==cfgs['Token']['arrival_offsets']
        table={a:lookup[a,m,o,b] for a in ('Layer','Token')}
        arrivals=cfgs['Layer']['arrival_offsets']
        fixed={a:predicted(arrivals,[a]*(count//18),table,b,32,o) for a in table}
        cells.append(dict(key=key,names={a:r['name'] for a,r in values.items()},table=table,
            arrivals=arrivals,fixed=fixed))
    losses=[]
    details=[]
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for kappa in (.25,.5,1.,2.,4.):
            token_regret=[];latency_regret=[]
            for cell in cells:
                stage,m,o,q,seed,count,b=cell['key']
                result=pressure_rule(cell['arrivals'],cell['table'],kappa,b,32,o)
                best_tokens=max(v['qualified_output_tokens_s'] for v in cell['fixed'].values())
                best_TTFT=min(v['TTFT_p95_ms'] for v in cell['fixed'].values())
                loss=(best_tokens-result['metrics']['qualified_output_tokens_s'])/max(best_tokens,1e-12)
                latency=result['metrics']['TTFT_p95_ms']/best_TTFT-1
                token_regret.append(loss);latency_regret.append(latency)
                details.append(dict(key=cell['key'],kappa=kappa,names=cell['names'],
                    fixed=cell['fixed'],rule=result,qualified_token_regret=loss,TTFT_p95_regret=latency))
            losses.append(dict(kappa=kappa,mean_qualified_token_regret=statistics.mean(token_regret),
                mean_TTFT_p95_regret=statistics.mean(latency_regret),cells=len(cells)))
        best=min(losses,key=lambda r:(r['mean_qualified_token_regret'],r['mean_TTFT_p95_regret'],r['kappa']))
        result=dict(selected_kappa=best['kappa'],training_cells=len(cells),losses=losses,
            selected_rule='SINGLE_PRESSURE_RATIO',protocol_SHA256=digest(ROOT/'state/minimal_knob_protocol_before_heldout.json'),
            cost_model_SHA256=digest(source),source_commit='4cf3f9b80a2a8618da3bdddb1140fab7f403c3c7',
            training_names=sorted({n for cell in cells for n in cell['names'].values()}),
            definition='R=(1+ready_D_count/b+calibrated_active_residual/reference_D_cohort_cost)/(1+pending_P_count). Select by calibrated queue-weighted P/D service demand.',
            evidence='OFFLINE_REPLAY',
            scope='Train only the five prespecified kappa values on historical discovery. No new held-out metric is read. One online pressure ratio plus fixed action metadata. This does not establish measured generalization, causal optimum or deployable cross-axis state handling.')
        target=ROOT/'configs/frozen_single_pressure_rule_v1.json'
        if target.exists():
            assert json.loads(target.read_text())['rule']==result
        else:
            write(target,dict(frozen_epoch=time.time(),rule=result))
        write(ROOT/'analysis/historical_single_pressure_training.json',dict(time=time.time(),losses=losses,details=details,
            evidence='OFFLINE_REPLAY',scope='Training performance only. Do not report it as held-out accuracy or measured Serving benefit.'))
    print('SINGLE_PRESSURE_RULE_FROZEN',best,flush=True)
    preserve()


if __name__=='__main__':
    main()
