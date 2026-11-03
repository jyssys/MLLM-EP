"""Measure larger exact local probes after an accepted Ouro joint screen."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .campaign import gate
from .analysis_lane import quiet_analysis
from .verify_long_probes import verify
from .d_graph_state_followup import warm_reference
from .independent_joint_repeats import captures


def main():
    while time.time() < LONG_START_CUTOFF:
        path = ROOT/'state/independent_joint_repeats.json'
        if path.exists() and json.loads(path.read_text()).get('stage') in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
            break
        time.sleep(5)
    path = ROOT/'state/independent_joint_repeats_results.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    source = next((r for r in rows if r.get('accepted_joint_screen')), None)
    if source is None:
        write(ROOT/'state/qualified_probe_curve.json', dict(stage='NO_ACCEPTED_SOURCE', time=time.time()))
        return
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    template = json.loads((ROOT/'configs'/(source['baseline']+'.json')).read_text())
    ledger = []
    for size in (32, 64, 128, 256):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(template)
        cfg.update(name=f'curve_ouro_tokens{size}_gate8', D_prefill_graph_rows=size,
            D_probe_warmup_tokens=size, verify_transfer=True, state_probe=True,
            phase='correctness', timeout=900)
        name = put(cfg, 8, cfg['arrival_rate'], split='confirmation')
        row = dict(model='ouro', tokens=size, gate=name)
        write(ROOT/'state/qualified_probe_curve.json', dict(stage='STATE_GATE', name=name, time=time.time()))
        if not execute(name) or not gate(name):
            row['status'] = 'BLOCKED_TRANSFER_OR_WORK'
            ledger.append(row)
            break
        row['state'] = warm_reference('ouro', name, f'curve_ouro_tokens{size}_native_state_reference')
        if not all(row['state'].get(k) for k in ('complete', 'state_exact', 'output_exact', 'native_gate')):
            row['status'] = 'BLOCKED_STATE'
            ledger.append(row)
            break
        cfg.update(name=f'curve_ouro_tokens{size}_baseline128', verify_transfer=False,
            state_probe=False, phase='final_repeat')
        baseline = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
        if not execute(baseline):
            row['status'] = 'FAILED_BASELINE'
            ledger.append(row)
            break
        probe = copy.deepcopy(cfg)
        probe.update(name=f'curve_ouro_tokens{size}_probe128', D_probe_tokens=size,
            D_probe_count=8, D_probe_interval_s=2.)
        candidate = put(probe, 128, cfg['arrival_rate'], split='confirmation')
        write(ROOT/'state/qualified_probe_curve.json', dict(stage='MATCHED_PROBE', name=candidate, time=time.time()))
        if not execute(candidate) or not evidence.compare(baseline, candidate)['passed']:
            row['status'] = 'BLOCKED_EXECUTION_OR_PARITY'
            ledger.append(row)
            break
        row['correctness'] = verify(candidate)
        with quiet_analysis(candidate+'_metrics'):
            row['baseline'], _, _ = serving.serving_one(cfg)
            row['probe'], _, _ = serving.serving_one(probe)
        row['measured_captures'] = {n: captures(n) for n in (baseline, candidate)}
        m = row['probe']
        row['whole_run_D_SLO_pass'] = bool(m['request_TPOT_ms_p95'] <= .9*m['TPOT_threshold_ms']
            and m['token_ITL_ms_p99'] <= m['ITL_threshold_ms'])
        row['accepted'] = bool(row['whole_run_D_SLO_pass'] and row['correctness'].get('exact')
            and row['correctness'].get('full_work') and not any(row['measured_captures'].values()))
        row['scope'] = 'One matched exploratory start. Independent local prompts, not migration or a joint P device proof at this probe size.'
        ledger.append(row)
        write(ROOT/'state/qualified_probe_curve_results.json', ledger)
        if not row['accepted']:
            break
    write(ROOT/'state/qualified_probe_curve_results.json', ledger)
    write(ROOT/'state/qualified_probe_curve.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/qualified_probe_curve.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
