"""Record the source-based selection before discovery timing."""
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    target=ROOT/'artifacts/dynamic_baseline_selection.json'
    assert not target.exists()
    write(target,dict(decision='INCUMBENT_CONTINUE',immutable=True,
        model=MODEL_ID,revision=REVISION,q=.5,min_depth=2,max_depth=4,kv='last_exited',
        CDB_revision=CDB_REVISION,gate='original checkpoint early_exit_gate',
        prefill='full R4',decode='live online cumulative gate. Physical loop skipping.',
        cache='96 depth-indexed K/V pairs. Copy terminal token KV to deeper slots before any later read.',
        causal_kv_value_rule='At depth d, past exited-token KV equals min(d, token_exit_depth). Dense copied storage, not compact Ragged page aliases.',
        source_selection_reason='The incumbent already runs the pinned CDB authors original-gate implementation. It is not a separate challenger.',
        candidates=[dict(name='CDB authors original gate',sha=CDB_REVISION,status='SAME_IMPLEMENTATION_AS_INCUMBENT',
            source='https://github.com/kschwethelm/continuous-depth-batching',license='MIT plus third-party notices'),
            dict(name='Enabling Dynamic Computation',status='UPSTREAM_CODE_UNAVAILABLE',
                source='https://arxiv.org/abs/2610.09013',
                scope='No author repository was verified through the arXiv record, paper links, and exact-title repository search on 2026-10-18. This is not proof of universal absence.'),
            dict(name='vllm-rlt',sha=RLT_REVISION,status='AUDITED_NOT_PORTED',
                source='https://github.com/ThinkFlowLab/vllm-rlt',
                reason='Independent optional runtime. No evidence of a better matched-quality original gate. Existing official CDB is runnable. Avoid another engine port.')],
        inherited_warning='QUALITY_WARNING; REFERENCE_MATCH_PARTIAL. No fixed-R4 quality equivalence.',
        numerical_reference='v2 teacher-aligned and physical exit-copy probes retained without modification.',
        confirmation='No better independent challenger. No selection-confirmation quality cohort is required.',
        new_quality='64 frozen IDs are being rechecked at a common 8192 cap. Do not count this as independent challenger evidence.',
        discovery_graphs=True,quality_graphs=False,synthetic_replay=False,
        graph_output_validation='Required before headline batching/scheduling claims.',
        models_not_trained=True,RL_not_trained=True))
    print('INCUMBENT_CONTINUE')

if __name__=='__main__':main()
