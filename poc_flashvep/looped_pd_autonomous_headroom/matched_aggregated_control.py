"""Compare native Mixed DP2 with PD after correctness and limited chunk tuning."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .analysis_lane import quiet_analysis


def main():
    # The primary same-window experiments have priority over this control.
    while time.time() < LONG_START_CUTOFF:
        files = [ROOT/'state'/n for n in ('independent_joint_repeats.json', 'huginn_window_validation.json')]
        if all(p.exists() and json.loads(p.read_text()).get('stage') in
               ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED') for p in files):
            break
        time.sleep(5)
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    ledger = []
    for model in ('ouro', 'huginn'):
        if time.time() >= LONG_START_CUTOFF:
            break
        base = json.loads((ROOT/'configs'/f'joint_confirm_{model}_s1_n128_baseline.json').read_text())
        base.update(warmup_count=3, timeout=1200, phase='final_repeat')
        base['native_transport_limits'] = dict(base.get('native_transport_limits', {}), shutdown_timeout=300)
        base.pop('D_probe_warmup_tokens', None)
        base.pop('D_prefill_graph_rows', None)
        rate = 5.378 if model == 'ouro' else 1.317
        reference = copy.deepcopy(base)
        reference.update(name=f'aggregated_{model}_PD_gate8', verify_transfer=True, state_probe=True, phase='correctness')
        ref = put(reference, 8, rate, split='test')
        if not execute(ref) or not evidence.full_work(ref)['passed']:
            ledger.append(dict(model=model, status='BLOCKED_PD_REFERENCE'))
            continue
        choices = []
        for chunk in (512, 2048):
            if time.time() >= LONG_START_CUTOFF:
                break
            dp = copy.deepcopy(base)
            dp.update(name=f'aggregated_{model}_C{chunk}_gate8', arm='DP2_NATIVE_MIXED',
                worker='native_mixed_worker', chunk=chunk, wavefront=True,
                capacity=8 if model == 'ouro' else 4, slots=8 if model == 'ouro' else 4,
                kv_blocks=16384 if model == 'ouro' else 22016,
                D_kv_blocks=16384 if model == 'ouro' else 22016,
                graph_rows=8 if model == 'ouro' else 4,
                colocated_prefill_graph_rows=base['P_prefill_graph_rows'],
                verify_transfer=False, state_probe=True, phase='correctness',
                transport_scope='Two colocated replicas. No P-to-D transfer.',
                memory_scope='Each replica pool supports its configured active-request limit. Pure-Prefill graph reuse is enabled for both execution families.')
            name = put(dp, 8, rate, split='test')
            okay = execute(name)
            parity = evidence.compare(ref, name) if okay else {}
            work = evidence.full_work(name) if okay else {}
            row = dict(model=model, chunk=chunk, gate=name, parity=parity, work=work)
            if not (okay and parity.get('passed') and work.get('passed')):
                row['status'] = 'BLOCKED_DP_CORRECTNESS'
                ledger.append(row)
                continue
            dp.update(name=f'aggregated_{model}_C{chunk}_calibration24', state_probe=False, phase='calibration')
            name = put(dp, 24, rate, split='calibration')
            if execute(name):
                with quiet_analysis(name+'_metrics'):
                    m, _, _ = serving.serving_one(dp)
                row['calibration'] = m
                choices.append((m['SLO_goodput'], -m['TTFT_ms_p95'], copy.deepcopy(dp)))
            ledger.append(row)
            write(ROOT/'state/matched_aggregated_results.json', ledger)
        if not choices:
            continue
        dp = max(choices, key=lambda x: (x[0], x[1]))[2]
        dp.update(name=f'aggregated_{model}_selected_gate32', phase='correctness', arrival_seed=2026111201)
        name = put(dp, 32, rate, split='confirmation')
        pd = copy.deepcopy(base)
        pd.update(name=f'aggregated_{model}_PD_gate32', phase='correctness', arrival_seed=2026111201)
        native = put(pd, 32, rate, split='confirmation')
        if not (execute(native) and execute(name) and evidence.compare(native, name)['passed']
                and evidence.full_work(name)['passed']):
            ledger.append(dict(model=model, status='BLOCKED_INDEPENDENT32'))
            continue
        write(ROOT/'state'/f'aggregated_{model}_frozen_choice.json', dict(time=time.time(), chunk=dp['chunk'],
            selection='Highest joint SLO-goodput on disjoint 24-request calibration. TTFT p95 breaks ties.',
            scope='Limited two-chunk tuning. Not a globally optimal production baseline.'))
        rates = (2.5, 5.378, 6.356) if model == 'ouro' else (.8, 1.317, 1.556)
        for index, offered in enumerate(rates):
            if time.time() >= LONG_START_CUTOFF:
                break
            names = {}
            record = dict(model=model, target_QPS=offered, chunk=dp['chunk'], evidence='MEASURED_ONE_START_PER_RATE')
            for arm in (('PD', 'DP') if index % 2 == 0 else ('DP', 'PD')):
                cfg = copy.deepcopy(base if arm == 'PD' else dp)
                cfg.update(name=f'aggregated_{model}_{arm}_rate{index}_128', phase='final_repeat',
                    independent_start=1, arrival_seed=2026111301)
                name = put(cfg, 128, offered, split='confirmation')
                write(ROOT/'state/matched_aggregated.json', dict(stage='RATE_CONTROL', name=name, time=time.time()))
                names[arm] = name
                record[arm+'_complete'] = execute(name)
                if record[arm+'_complete']:
                    with quiet_analysis(name+'_metrics'):
                        record[arm], _, _ = serving.serving_one(cfg)
            if all(record.get(a+'_complete') for a in ('PD', 'DP')):
                record['paired_output_parity'] = evidence.compare(names['PD'], names['DP'])
            ledger.append(record)
            write(ROOT/'state/matched_aggregated_results.json', ledger)
    write(ROOT/'state/matched_aggregated_results.json', ledger)
    write(ROOT/'state/matched_aggregated.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/matched_aggregated.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
