"""Revalidate every exact layer cut in the new evidence namespace."""
import fcntl
import json
from .common import ROOT, SLO_PRIOR, QUIET_LOCK, resources, save, write
from .run import execute


def gate(name, output_path=None):
    from ..dense_ar_slo_layer_pd import stages, generic_qualification
    stages.ROOT = ROOT
    work = stages.state_gate(name, 32)
    expected = generic_qualification.independent()
    assert len(expected) == 32
    folder = ROOT / 'raw' / name
    src = json.loads((folder / 'full_prompt_state_prefill.json').read_text())['requests']
    dst = json.loads((folder / 'full_prompt_state_decode.json').read_text())['requests']
    checks = {}
    for rid, ref in expected.items():
        state = src[rid] if work['cuts'][rid] == 36 else dst[rid]
        checks[rid] = dict(KV=state['KV'] == ref['KV'],
            hidden=state['final_hidden_SHA256'] == ref['final_hidden_SHA256'])
    from ..duetserve_agg_decode import analyze
    analyze.ROOT = ROOT
    output = analyze.output_gate(name)
    passed = work['passed'] and output['passed'] and len(checks) == 32 and all(all(c.values()) for c in checks.values())
    value = dict(passed=passed, work=work, output=output, checks=checks,
        actual_offloaded_layer_set=sorted({36-cut for cut in work['cuts'].values()}),
        evidence='CORRECTNESS_ONLY', scope='Check only the actual cuts recorded in work.cuts. Validate their P/D visits, full36 prompt KV, final hidden, native import, ACK and normal lifetimes. This gate does not prove unexecuted cuts or online policy benefit.')
    write(output_path or ROOT / 'analysis' / (name + '_full_state_gate.json'), value)
    assert passed, value
    return value


def main():
    cfg = json.loads((SLO_PRIOR / 'configs/qual_coalesced_all_cuts_N32_full_state_v1.json').read_text())
    cfg.update(name='Layer_all_cuts_state_N32_v1', entry_module='poc_flashvep.dense_ar_layer_token_pd.dynamic_serving',
        D_admission_batch=4, D_wait_for_source_macro=True, preplanned_source_groups=True,
        warmup_batch_sizes=[1, 2, 4], warmup_count=2, count=32,
        arrival_offsets=[0.] * 32, request_caps=[2] * 32, ar_cudagraph_mode='FULL_DECODE_ONLY',
        forced_layers=[0, 2, 4, 8, 12, 0, 8, 12])
    try:
        assert execute(save(cfg))
        from ..dense_ar_slo_layer_pd import stages, generic_qualification
        stages.ROOT = ROOT
        work = stages.state_gate(cfg['name'], 32)
        assert work['passed'], work
        expected = generic_qualification.independent()
        assert len(expected) == 32
        folder = ROOT / 'raw' / cfg['name']
        src = json.loads((folder / 'full_prompt_state_prefill.json').read_text())['requests']
        dst = json.loads((folder / 'full_prompt_state_decode.json').read_text())['requests']
        checks = {}
        for rid, ref in expected.items():
            state = src[rid] if work['cuts'][rid] == 36 else dst[rid]
            checks[rid] = dict(KV=state['KV'] == ref['KV'],
                hidden=state['final_hidden_SHA256'] == ref['final_hidden_SHA256'])
        from ..duetserve_agg_decode import analyze
        analyze.ROOT = ROOT
        output = analyze.output_gate(cfg['name'])
        passed = work['passed'] and output['passed'] and len(checks) == 32 and all(all(c.values()) for c in checks.values())
        write(ROOT / 'analysis/Layer_all_cuts_N32_full_state_gate.json', dict(passed=passed,
            work=work, output=output, checks=checks, evidence='CORRECTNESS_ONLY',
            scope='All0/2/4/8/12 actual P/D visits, valid all36 prompt KV, final hidden, native import, ACK and normal lifetimes. Forced cuts are not an online selection result.'))
        assert passed
        print('LAYER_ALL_CUTS_FULL_STATE_PASS', flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
