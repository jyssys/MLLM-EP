"""Allow compatible complete source groups to form a real destination batch."""
import fcntl,collections
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .stages import state_gate
from .analyze import records


def main():
    cfg=base('qual_combine_ready_macros_Last8_N16_v1','FORCED_QUALIFICATION',count=16,
        cap=16,request_caps=[2]*16,arrival_offsets=[0.]*16,arrival_rate=None,
        P_group_size=1,D_wait_for_source_macro=True,D_combine_ready_macros=True,
        D_admission_batch=8,D_cohort_collect_ms=2000,hidden_source_slots=32,
        forced_layers=[8],negotiation_timeout_s=120,verified_ByteLevel_priming=True,
        direct_phase_norm=True,nixl_manual_coalesce=True,nixl_descriptor_sort=False,
        verify_layer_execution=True,hidden_verify_hash=True,full_prompt_state_witness=True,
        full_prompt_state_witness_roles=['prefill','decode'],correctness_only=True,clean_timing=False,
        phase='CROSS_SOURCE_GROUP_BATCH_CORRECTNESS_WITH_ARTIFICIAL_COLLECTION_NOT_PERFORMANCE')
    try:
        name=save(cfg);assert execute(name,restore=False)
        assert against_previous(name)['passed'];assert state_gate(name,16)['passed']
        f=ROOT/'raw'/name;expected=independent()
        actual=json.loads((f/'full_prompt_state_decode.json').read_text())['requests']
        checks={rid:dict(KV_equal=v['KV']==expected[rid]['KV'],
            Hidden_equal=v['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256']) for rid,v in actual.items()}
        groups=collections.defaultdict(set)
        for row in records(f/'D_host_admission.jsonl'):groups[row['cohort']].add(row['source_group'])
        combined=any(len(s)>1 for s in groups.values())
        passed=len(checks)==16 and all(all(v.values()) for v in checks.values()) and combined
        write(ROOT/'analysis/combined_ready_macros_full_state_gate.json',dict(name=name,checks=checks,
            actual_source_groups_per_D_cohort={str(k):sorted(v) for k,v in groups.items()},
            passed=passed,evidence='CORRECTNESS_ONLY',
            scope='All work was actually Source-ready. The artificial collection delay establishes batching correctness only. It is not a Serving performance result.'))
        assert passed
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
