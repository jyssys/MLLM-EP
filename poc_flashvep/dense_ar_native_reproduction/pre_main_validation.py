"""Require explicit reference and accounting gates before a Main sweep."""
import json
import time
from .common import ROOT,write


def main():
    state=json.loads((ROOT/'analysis/official_invariant_N8_qualification_v1.json').read_text())
    n32={method:json.loads((ROOT/'analysis'/f'qual_{method}_LongBench_N32_B8192_S1024_BI1_q2_v1_output_gate.json').read_text())['passed'] for method in ('Native','Agg')}
    rows=json.loads((ROOT/'analysis/clean_Native_standard_decode_comparison_v1.json').read_text())['rows']
    reference=rows[0]['metrics']['mean_TBT_ms']
    ratios={r['name']:r['metrics']['mean_TBT_ms']/reference for r in rows}
    profile=json.loads((ROOT/'analysis/decode_kernel_host_breakdown_v2.json').read_text())['results']['BI1']
    passed=state['passed'] and all(n32.values()) and len(rows)==3 and all(r['strict_output_parity'] for r in rows) and profile['observer_gate_passed']
    write(ROOT/'analysis/Native_reproduction_pre_main_gate.json',dict(passed=passed,time=time.time(),
        N8_full_state=state['passed'],N32_output=n32,diagnostic_C1_decode_ratios=ratios,
        profile_observer_gate_passed=profile['observer_gate_passed'],
        numerical_family='OFFICIAL_BATCH_INVARIANT',Native_protocol='OFFICIAL_PROXY',Native_transport='STANDARD_NIXL',
        Model_revision='b968826d9c46dd6066d109eabc6255188de91218',
        qualification='REFERENCE_REPRODUCTION_ONLY_NOT_DECODE_SLO_OR_HEADROOM_PASS',
        scope='N8 full state and N32 natural outputs pass. C1 latency ratios are reported descriptively, with no newly tuned agreement threshold and no5% noninferiority CI claim. Standard NIXL concurrent-import stalls remain measured limitations. Coalescing is an ablation, not an unproven Primary improvement. No Layer/Token resumption is authorized by this gate.'))
    assert passed,'Do not start Main from an unqualified model/runtime comparison'
    print('PRE_MAIN_REFERENCE_GATE_PASS',flush=True)


if __name__=='__main__':main()
