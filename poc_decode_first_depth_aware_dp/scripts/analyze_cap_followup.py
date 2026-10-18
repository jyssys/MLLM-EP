"""Score both longer-cap policies. Never overwrite the 4096-token primary result."""
import statistics
from poc_decode_first_depth_aware_dp.contract import *
from poc_decode_first_depth_aware_dp.scripts.quality_analyze import interval
from poc_async_recurrent_pp.scoring import score


def main():
    output=ROOT/'artifacts/quality_cap_sensitivity.json'
    assert not output.exists()
    controller=read(ROOT/'artifacts/quality_cap_followup_controller.json')
    assert controller['all_success']
    wanted=set(controller['selected_ids'])
    protocol={r['id']:r for r in read(ROOT/'artifacts/quality_protocol_v3_pilot32.json')['rows']}
    primary=read(ROOT/'artifacts/quality_pilot_analysis_v2/quality_predictions.json')
    updated={(r['id'],r['policy']):r for r in primary};observed={};comparisons=[]
    primary_completion=[]
    for dataset in ['gsm8k','math500']:
        ids=sorted({r['id'] for r in primary if r['dataset']==dataset})
        fixed=[updated[i,'fixed'] for i in ids];dynamic=[updated[i,'dynamic'] for i in ids]
        f=[int(r['answer_matches'] and not r['truncated']) for r in fixed]
        d=[int(r['answer_matches'] and not r['truncated']) for r in dynamic]
        loss=[100*(x-y) for x,y in zip(f,d)]
        primary_completion.append(dict(dataset=dataset,n=len(ids),fixed_correct=sum(f),dynamic_correct=sum(d),
            accuracy_drop_pp=statistics.mean(loss),paired_bootstrap_CI95_drop_pp=interval(loss)))
    for folder in sorted((ROOT/'outputs').glob('quality_*_cap8192_v1')):
        assert read(folder/'status.json')['success']
        for path in sorted(folder.glob('*.json')):
            raw=read(path)
            if not isinstance(raw,dict) or 'depth_events' not in raw:continue
            key=raw['id'],raw['policy'];assert raw['id'] in wanted and key not in observed
            result=score(protocol[raw['id']],dict(text=raw['text'],finish_reason='eos' if raw['finish_reason'] in ['eos','stop'] else 'length'))
            prior=updated[key];prior_raw=read(prior['raw_path'])
            prefixes_equal=raw['output_ids'][:len(prior_raw['output_ids'])]==prior_raw['output_ids']
            depth_prefix_equal=([e['exit_depth_executed'] for e in raw['depth_events'][:len(prior_raw['depth_events'])]]==
                [e['exit_depth_executed'] for e in prior_raw['depth_events']])
            comparisons.append(dict(id=key[0],policy=key[1],primary_output_tokens=prior_raw['output_length'],
                followup_output_tokens=raw['output_length'],followup_finish=raw['finish_reason'],
                primary_answer_matches=prior['answer_matches'],followup_answer_matches=result['answer_matches'],
                extracted_answer=result['extracted_answer'],gold=protocol[key[0]]['answer'],
                primary_token_prefix_equal=prefixes_equal,primary_depth_prefix_equal=depth_prefix_equal,
                raw_path=str(path),followup_truncated=raw['finish_reason']=='length'))
            updated[key]=dict(prior,answer_matches=result['answer_matches'],truncated=raw['finish_reason']=='length')
            observed[key]=raw
    assert set(observed)=={(i,p) for i in wanted for p in ['fixed','dynamic']}
    summaries=[]
    for dataset in ['gsm8k','math500']:
        ids=sorted({i for i,p in updated if protocol[i]['dataset']==dataset})
        fixed=[updated[i,'fixed'] for i in ids];dynamic=[updated[i,'dynamic'] for i in ids]
        loss=[100*(int(f['answer_matches'])-int(d['answer_matches'])) for f,d in zip(fixed,dynamic)]
        summaries.append(dict(dataset=dataset,n=len(ids),fixed_correct=sum(r['answer_matches'] for r in fixed),
            dynamic_correct=sum(r['answer_matches'] for r in dynamic),accuracy_drop_pp=statistics.mean(loss),
            paired_bootstrap_CI95_drop_pp=interval(loss),fixed_truncations=sum(r['truncated'] for r in fixed),
            dynamic_truncations=sum(r['truncated'] for r in dynamic),
            fixed_completed_correct=sum(r['answer_matches'] and not r['truncated'] for r in fixed),
            dynamic_completed_correct=sum(r['answer_matches'] and not r['truncated'] for r in dynamic)))
    write(output,dict(rows=summaries,comparisons=comparisons,primary_completion_aware=primary_completion,
        all_primary_token_prefixes_equal=all(r['primary_token_prefix_equal'] for r in comparisons),
        all_primary_depth_prefixes_equal=all(r['primary_depth_prefix_equal'] for r in comparisons),
        primary_analysis_preserved=True,subsequent_stages_allowed=False,
        scope='Secondary paired cap sensitivity. Only the primary truncation union uses cap 8192; all other questions retain primary outputs.'))
    print('CAP_SENSITIVITY',summaries)


if __name__=='__main__':main()
