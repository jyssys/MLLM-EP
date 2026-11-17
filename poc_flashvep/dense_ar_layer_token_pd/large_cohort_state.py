"""Check the full-prompt Last12 continuation in an actual D batch8."""
import fcntl
import json
from .common import ROOT,QUIET_LOCK,resources,save,write
from .axis_configuration import base
from .run import execute
from .layer_qualification import gate


def main():
    corrections=[]
    for path in (ROOT/'analysis').glob('*_full_state_gate.json'):
        row=json.loads(path.read_text());cuts=row.get('work',{}).get('cuts',{})
        actual=sorted({36-cut for cut in cuts.values()})
        if actual and 'All0/2/4/8/12 actual' in row.get('scope','') and actual!=[0,2,4,8,12]:
            corrections.append(dict(original_gate=path.name,actual_offloaded_layer_set=actual,
                corrected_scope='The gate validates only the executed cuts in work.cuts. It does not validate missing cut sizes. Original hashes, checks and raw evidence remain unchanged.'))
    write(ROOT/'analysis/state_gate_scope_corrections.json',dict(corrections=corrections,
        evidence='TRACE_ANALYSIS',scope='Correct an over-general scope sentence in an early derived gate. Do not rewrite original gates or raw evidence. The full all-cut tests remain separate.'))
    try:
        cfg=base('Layer12_actualD8_state_N32_v1',method='Static',count=32,cap=2,layers=12,
            request_caps=[2]*32,arrival_offsets=[0.]*32,matched=True,
            P_group_size=8,P_slots=4,D_admission_batch=8,
            correctness_only=True,clean_timing=False,runtime_observation=True,
            verify_layer_execution=True,full_prompt_state_witness=True,
            full_prompt_state_witness_roles=['prefill','decode'],hidden_verify_hash=True,
            verify_native_prefix_KV=True,warmup_count=0,warmup_batch_sizes=[],
            hidden_source_slots=32,hidden_destination_slots=32,negotiation_timeout_s=120.)
        # Output2 is a state-only witness. Frozen serving SLOs have no cap2.
        # Reuse a completed GPU receipt without running the latency analyzer.
        assert execute(save(cfg));value=gate(cfg['name'])
        measured=json.loads((ROOT/'raw'/cfg['name']/'measured_geometry_decode.json').read_text())['counts']
        assert any(json.loads(key)['requests']==8 and json.loads(key)['prefilling'] for key in measured)
        write(ROOT/'analysis/Layer12_actualD8_full_state_qualification.json',dict(passed=value['passed'],
            gate=cfg['name']+'_full_state_gate.json',actual_D_prompt_batch8=True,
            evidence='CORRECTNESS_ONLY',scope='Source macro8 executes with P_slots4. Destination actually executes prefill batch8. All32 valid prompt KV and final hidden match the independent full-forward reference.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
