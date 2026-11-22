"""Require all8 original prompts per action to match the exact boundary teacher."""
import collections
import json
import torch
from .common import ROOT,write
from .qualification import audit


def main():
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/'configs/family_full_state_qualification_frozen_v1.json').read_text())
    teacher=ROOT/'raw/Q7_Full36_BoundaryTeachers_v1';actual=ROOT/'raw/Q7_Hybrid_Family208_FullState_v1'
    for folder in (teacher,actual):assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    expected=json.loads((teacher/'prompt_state_single.json').read_text())['requests']
    measured=json.loads((actual/'axis_events_decode.json').read_text())['states']
    gold={r['sample_id']:r for r in map(json.loads,(teacher/'requests.jsonl').read_text().splitlines())}
    output={r['sample_id']:r for r in map(json.loads,(actual/'requests.jsonl').read_text().splitlines())}
    coverage=audit(actual.name);rows=[];groups=collections.defaultdict(list)
    for item in manifest['mapping']:
        i,j=item['actual_index'],item['teacher_index'];key='q'+str(i);target='q'+str(j)
        hidden=torch.load(teacher/f'numerical_single_{target}.pt',map_location='cpu',weights_only=True)['hidden_last']
        state=measured[key];observed=torch.load(state['tensor_file'],map_location='cpu',weights_only=False)['hidden'][-1]
        row=dict(**item,full_valid_KV_exact=state['KV']==expected[target]['KV'],
            last_hidden_exact=torch.equal(hidden,observed),greedy16_exact=output[i]['output_ids']==gold[j]['output_ids'])
        row['passed']=row['full_valid_KV_exact'] and row['last_hidden_exact'] and row['greedy16_exact']
        rows.append(row);groups[json.dumps(item['action'],sort_keys=True)].append(row)
    actions=[]
    for key,values in groups.items():
        assert len(values)==8
        action=json.loads(key);actions.append(dict(axis=action['kind'],cut=action['cut'],
            suffix_fraction=action.get('suffix_fraction') if action['kind']=='Hybrid' else None))
    passed=all(r['passed'] for r in rows) and coverage['passed']
    result=dict(passed=passed,rows=rows,action_count=len(actions),original_request_count=8,executions=len(rows),
        teacher_server_starts=1,partition_server_starts=1,coverage=coverage,evidence='CORRECTNESS_ONLY',
        scope='Each action has8 original calibration prompts. All36 full valid KV hashes,last hidden and16 greedy tokens are compared. '
            'The repeated action executions are not independent performance starts. No C3 or serving-gain claim follows.')
    write(ROOT/'analysis/family_q7_full_state_comparisons.json',result)
    write(ROOT/'state/C0_family_q7_gate.json',dict(passed=passed,qualified_axes=sorted({a['axis'] for a in actions}) if passed else [],
        qualified_actions=actions if passed else [],mixed_rows_passed=False,n_per_action=8,action_count=len(actions),
        proof='analysis/family_q7_full_state_comparisons.json',evidence='CORRECTNESS_ONLY'))
    print('Family Q7 full-state gate',passed,'executions',len(rows),flush=True)
    assert passed,'Preserve mismatches. Do not use a failed family for performance.'


if __name__=='__main__':main()
