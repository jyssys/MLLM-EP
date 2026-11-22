"""Qualify additional static actions and heterogeneous mixed-row actions."""
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .next_jobs import plan


def main():
    assert json.loads((ROOT/'state/GPU_digest_q9_gate.json').read_text())['passed']
    capsule,sha=freeze('continuous_extra_action_qualification_v1_20261121')
    lanes=[[],[]];actions=[]
    for cut in (2,8,12):
        for axis in ('Layer','Token'):actions.append(dict(axis=axis,cut=cut))
    actions.extend(dict(axis='Hybrid',cut=m,fraction=f) for m,f in ((2,.125),(8,1/16),(12,1/16)))
    common=dict(qps=8,seed=2026112271,correctness=True,
        axis_state_capture=True,axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,
        gpu_state_digest=True,continuous_source_credit_limit=4,
        source_capsule=capsule,source_capsule_SHA256=sha,wall_timeout=1800,serving_timeout=1500)
    for index,action in enumerate(actions):
        lane=index%2;first=(2,6)[lane];axis=action['axis'];cut=action['cut'];extra={}
        if axis=='Hybrid':extra=dict(hybrid_suffix_fraction=action['fraction'],partition_aligned_chunks=True)
        name=f"C3Q10_{axis}{cut}_f{action.get('fraction',0)}_Natural32_v1"
        lanes[lane].append(create(name,axis,'continuous_qualification32_natural.json',cut=cut,
            pair=(first,first+1),**common,**extra))
    family=json.loads((ROOT/'configs/family_full_state_qualification_frozen_v1.json').read_text())['actions']
    rows=json.loads((ROOT/'data/continuous_qualification32_natural.json').read_text())
    action_map={'q'+str(i):family[i%len(family)] for i in range(len(rows))}
    lanes[1].append(create('C3Q10_HeterogeneousFamily_Natural32_v1','Hybrid',
        'continuous_qualification32_natural.json',cut=12,pair=(6,7),
        hybrid_suffix_fraction=.125,partition_aligned_chunks=True,
        axis_actions_by_request=action_map,**common))
    write(ROOT/'configs/continuous_extra_actions_qualification_v1.json',dict(
        actions=actions,heterogeneous_actions=action_map,source_capsule=capsule,
        source_capsule_SHA256=sha,evidence='FROZEN_CORRECTNESS_DESIGN',
        scope='Natural32 tests are correctness and task-quality checks,not performance starts. '
            'The heterogeneous test does not implement a predictor or a dynamic performance policy. '
            'Each performance action still needs its own full-state and mixed-row proof.'))
    for lane,jobs in enumerate(lanes):plan(f'continuous_extra_action_qualification_lane{lane}_v1',jobs)
    print('Frozen additional static and heterogeneous Natural32 qualifications.',flush=True)


if __name__=='__main__':main()
