"""Confirm all layer cuts on128 unseen prompts without timing claims."""
import fcntl
import json
from .common import ROOT, QUIET_LOCK, resources, save, write, bind_previous
from .run import execute
from .fresh_reference import REFERENCE


def main():
    cfg = json.loads((ROOT / 'configs/Layer_all_cuts_state_N32_v1.json').read_text())
    name = 'unseen_Layer_all_cuts_state_N128_v1'
    cfg.update(name=name, count=128, dataset='heldout128_8000.json', reference_name=REFERENCE,
        request_caps=[2]*128, arrival_offsets=[0.]*128, timeout=1200, wall_timeout=1600)
    try:
        assert execute(save(cfg))
        bind_previous()
        from ..dense_ar_slo_layer_pd import stages
        stages.ROOT = ROOT
        work = stages.state_gate(name,128)
        assert work['passed']
        expected = json.loads((ROOT / 'raw' / REFERENCE / 'full_prompt_state_prefill.json').read_text())['requests']
        folder = ROOT / 'raw' / name
        src = json.loads((folder / 'full_prompt_state_prefill.json').read_text())['requests']
        dst = json.loads((folder / 'full_prompt_state_decode.json').read_text())['requests']
        checks = {}
        for rid, ref in expected.items():
            value = src[rid] if work['cuts'][rid]==36 else dst[rid]
            checks[rid] = dict(KV=value['KV']==ref['KV'], hidden=value['final_hidden_SHA256']==ref['final_hidden_SHA256'])
        from ..duetserve_agg_decode import analyze
        analyze.ROOT = ROOT
        output = analyze.output_gate(name)
        passed = len(checks)==128 and all(all(v.values()) for v in checks.values()) and output['passed'] and work['passed']
        write(ROOT / 'analysis/Layer_N128_full_state_gate.json', dict(passed=passed,
            checks=checks, output=output, work=work, evidence='CORRECTNESS_ONLY',
            scope='All0/2/4/8/12 forced cuts on128 unseen real prompts. Actual P prefix, D suffix and Full36 Decode visits. Full valid prompt KV and full final hidden hashes. Normal lifetime and ACK. These instrumented latencies are not performance data.'))
        assert passed
        print('LAYER_N128_FULL_STATE_PASS', flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
