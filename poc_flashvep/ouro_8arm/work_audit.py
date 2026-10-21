"""Audit executed work counters against completed output and exit traces."""
import collections
from .contract import *
from .serving_analysis import table


def main():
    rows=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.name.startswith(('quality_','extension400_','matched_natural_','observer_control_','profile_live_')):continue
        answers=collections.defaultdict(list);ranks=collections.defaultdict(list)
        for file in folder.glob('*.json'):
            r=read(file)
            if not isinstance(r,dict):continue
            if r.get('id') and 'depths' in r:answers[r['cohort_index']].append(r)
            if file.name.startswith('cohort_') and 'rank' in r:ranks[int(file.name.split('_')[1])].append(r)
        for index,rr in ranks.items():
            actual=answers[index]
            if not actual:continue
            arm=actual[0]['arm']
            if len(rr)!=len(ARMS[arm]['GPUs']):continue
            decode_work=sum(sum(r['depths'][1:]) for r in actual)
            expected=24*(4*len(actual)+decode_work)
            token_expected=24*(4*sum(len(r['prompt_ids']) for r in actual)+decode_work)
            counters=[r['counts'] for r in rr if 'counts' in r]
            if counters:
                observed=sum(c.get('physical_layer_invocations',0) for c in counters)
                token_observed=sum(c.get('token_layer_work',0) for c in counters)
                assert observed==expected,(folder,index,observed,expected)
                assert token_observed==token_expected,(folder,index,token_observed,token_expected)
                scope='Independent stage-body invocation counters compared with completed token-depth traces. Setup excluded.'
            else:
                observed=None;token_observed=None
                for r in rr:
                    if 'forward_loop_invocations' in r:assert r['forward_loop_invocations']==4*len(actual)+decode_work
                scope='TP aggregate trace accounting only. Independent GPU Graph-launch evidence requires the heavy trace.'
            rows.append(dict(run=folder.name,arm=arm,cohort=index,N=len(actual),decode_tokens=sum(len(r['depths'])-1 for r in actual),
                executed_decode_loop_work=decode_work,fixed_R4_decode_loop_work=4*sum(len(r['depths'])-1 for r in actual),
                expected_layer_calls=expected,observed_stage_layer_calls=observed,expected_token_layer_work=token_expected,
                observed_token_layer_work=token_observed,scope=scope))
    table(ROOT/'executed_work_audit.csv',rows)
    print('WORK_COUNTER_COHORTS_PASS',len(rows),flush=True)


if __name__=='__main__':main()
