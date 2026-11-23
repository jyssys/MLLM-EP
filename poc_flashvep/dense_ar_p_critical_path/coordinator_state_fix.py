"""Read distributed strict teachers without changing the numerical gate."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind

PLAN='coordinator_same_source_state_schema_v4.json'


def compare(control,candidate):
    import torch
    from .critical_analysis import records
    from ..dense_ar_longbench400_pd.qualification_gate import KV_equal
    folders=[ROOT/'raw'/name for name in (control,candidate)]
    states=[json.loads((folder/'axis_events_decode.json').read_text())['states'] for folder in folders]
    outputs=[{r['sample_id']:r for r in records(folder/'requests.jsonl')} for folder in folders]
    assert set(outputs[0])==set(outputs[1])
    rows=[]
    for index in sorted(outputs[0]):
        rid='q'+str(index);a,b=[state[rid] for state in states]
        tensors=[torch.load(state['tensor_file'],map_location='cpu',weights_only=False)['hidden'] for state in (a,b)]
        layers=[{row['layer']:row for row in state['KV']} for state in (a,b)]
        assert set(layers[0])==set(layers[1])
        different=[layer for layer in sorted(layers[0]) if layers[0][layer]!=layers[1][layer]]
        rows.append(dict(request_id=rid,KV_exact=KV_equal(a['KV'],b['KV']),
            hidden_exact=torch.equal(*tensors),greedy_exact=outputs[0][index]['output_ids']==outputs[1][index]['output_ids'],
            first_different_KV_layer=different[0] if different else None))
    result=dict(control=control,candidate=candidate,rows=rows,
        passed=all(all(row[key] for key in ('KV_exact','hidden_exact','greedy_exact')) for row in rows),
        teacher_schema='axis_events_decode.json',candidate_schema='axis_events_decode.json',
        scope='Same-source distributed strict differential. Compare full captured hidden tensors,all valid KV hashes and greedy token IDs. '
            'Only the reader schema changes. Keep the earlier missing-file failure and every actual numerical failure. '
            'A failed strict result blocks this composition. It is not waived as BI0 variation.')
    write(ROOT/'analysis'/(candidate+'_same_source_differential.json'),dict(time=time.time(),**result))
    return result


def prepare():
    bind();parent=ROOT/'configs/coordinator_hygiene_qualification_r2.json'
    value=json.loads(parent.read_text());items=[]
    for old in value['items']:
        if not old['teacher']:continue
        item=dict(old);item['gate']='pc_coord_'+old['label'].lower()+'_composition_same_source_v4.json'
        items.append(item)
    target=ROOT/'configs'/PLAN;assert not target.exists()
    write(target,dict(time=time.time(),parent_plan_SHA256=digest(parent),items=items,
        source_capsule_SHA256=None,state_prefix='pc_coordinator_same_source_schema_v4',
        scope='Repair the CPU teacher-state reader. Reuse immutable completed strict GPU runs. '
            'Do not rerun a failed strict pair. New Natural32/400 runs are allowed only after the exact pair passes. '
            'Keep all R2 missing-schema failures and the new true state comparison.'))
    log=ROOT/'logs/coordinator_same_source_schema_v4.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.coordinator_state_fix','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/coordinator_same_source_schema_controller_v4.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    bind()
    from ..dense_ar_longbench400_pd import qualification_gate
    from ..dense_ar_output_aware_v2.c3_family import run as qualify
    previous=qualification_gate.compare;qualification_gate.compare=compare
    try:qualify(1,PLAN)
    finally:qualification_gate.compare=previous


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)
