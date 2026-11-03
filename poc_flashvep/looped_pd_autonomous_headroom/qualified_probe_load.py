"""Bound the frequency of exact D-local work after the size and Huginn gates."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .analysis_lane import quiet_analysis
from .verify_long_probes import verify
from .independent_joint_repeats import captures


def main():
    terminal = {'COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED', 'NO_D_SLO_CANDIDATE'}
    while time.time() < LONG_START_CUTOFF:
        paths = [ROOT/'state'/(name+'.json') for name in ('qualified_probe_curve', 'huginn_shorter')]
        if all(p.exists() and json.loads(p.read_text()).get('stage') in terminal for p in paths):
            break
        time.sleep(5)
    rows = json.loads((ROOT/'state/qualified_probe_curve_results.json').read_text())
    safe = [r for r in rows if r.get('accepted')]
    if not safe:
        write(ROOT/'state/qualified_probe_load.json', dict(stage='NO_ACCEPTED_SOURCE', time=time.time()))
        return
    source = max(safe, key=lambda row: row['tokens'])
    base = json.loads((ROOT/'configs'/(source['baseline']['name']+'.json')).read_text())
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    ledger = []
    for interval in (1., .5, .25):
        if time.time() >= LONG_START_CUTOFF:
            break
        tag = str(int(1000*interval))+'ms'
        cfg = copy.deepcopy(base)
        cfg.update(name=f'load_ouro_tokens{source["tokens"]}_{tag}_baseline128')
        baseline = put(cfg, 128, cfg['arrival_rate'], 'confirmation')
        write(ROOT/'state/qualified_probe_load.json', dict(stage='MATCHED_BASELINE', name=baseline, time=time.time()))
        if not execute(baseline):
            ledger.append(dict(name=baseline, status='BASELINE_FAILED'))
            break
        probe = copy.deepcopy(cfg)
        probe.update(name=f'load_ouro_tokens{source["tokens"]}_{tag}_probe128',
            D_probe_tokens=source['tokens'], D_probe_count=128, D_probe_interval_s=interval)
        candidate = put(probe, 128, cfg['arrival_rate'], 'confirmation')
        write(ROOT/'state/qualified_probe_load.json', dict(stage='MATCHED_PROBE', name=candidate, time=time.time()))
        row = dict(name=candidate, tokens=source['tokens'], requested_interval_s=interval,
            scope='One matched start. Actual completed probe rate is measured separately. No P kernel capture in this frequency control.')
        if not execute(candidate):
            row['status'] = 'PROBE_FAILED'
            ledger.append(row)
            break
        row['parity'] = evidence.compare(baseline, candidate)
        row['correctness'] = verify(candidate)
        with quiet_analysis(candidate+'_metrics'):
            row['baseline'], _, _ = serving.serving_one(cfg)
            row['probe'], _, _ = serving.serving_one(probe)
        records = [json.loads(line) for line in (ROOT/'raw'/candidate/'D_probe_requests.jsonl').read_text().splitlines()]
        row['completed_probes'] = len(records)
        row['additional_prefill_recurrent_rows'] = len(records)*source['tokens']*4
        row['additional_prompt_tokens_per_response_wall_s'] = len(records)*source['tokens']/row['probe']['elapsed_s']
        row['measured_captures'] = {name: captures(name) for name in (baseline, candidate)}
        m = row['probe']
        row['accepted'] = bool(row['parity']['passed'] and row['correctness'].get('exact')
            and row['correctness'].get('full_work') and not any(row['measured_captures'].values())
            and m['request_TPOT_ms_p95'] <= .9*m['TPOT_threshold_ms']
            and m['token_ITL_ms_p99'] <= m['ITL_threshold_ms'])
        ledger.append(row)
        write(ROOT/'state/qualified_probe_load_results.json', ledger)
        if not row['accepted']:
            break
    write(ROOT/'state/qualified_probe_load_results.json', ledger)
    write(ROOT/'state/qualified_probe_load.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/qualified_probe_load.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
