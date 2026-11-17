"""Separate additive host boundaries from empirical stream/compute models."""
import collections
import fcntl
import json
import statistics
from .common import QUIET_LOCK, ROOT, resources, write
from .analysis import records
from ..dense_ar_slo_layer_pd.analyze import percentile


def summary(values):
    return dict(count=len(values), mean=statistics.mean(values), p50=statistics.median(values),
                p95=percentile(values, .95), maximum=max(values)) if values else dict(count=0)


def fit(version='v2'):
    frozen=ROOT/'configs'/('empirical_host_cost_model_discovery_'+version+'.json')
    if frozen.exists():
        print('REUSE_IMMUTABLE_HOST_COST_MODEL',version,flush=True)
        return json.loads(frozen.read_text())['tables']
    cells = {kind: collections.defaultdict(list) for kind in ('P_source_ms', 'D_prepare_ms', 'negotiation_ms', 'decode_step_ms')}
    origins = {kind: collections.defaultdict(set) for kind in cells}
    decompositions = []
    for path in sorted((ROOT / 'raw').glob('*/dispatcher_result.json')):
        if not json.loads(path.read_text()).get('complete'):
            continue
        folder = path.parent
        cfg = json.loads((folder / 'configuration.json').read_text())
        if 'heldout' in cfg.get('dataset', '').lower() or cfg.get('phase', '').startswith('FROZEN_HELDOUT'):
            continue
        if cfg.get('correctness_only') or cfg.get('execution_events') or cfg.get('token_axis') or cfg.get('arm') == 'DP2':
            continue
        if cfg.get('evaluation_arm') == 'PChunk' or cfg.get('atomic_prompt') is False or cfg.get('credit_fast_lane'):
            continue
        if cfg.get('native_continuous_D_ingress'):
            continue
        rows = {'q' + str(r['sample_id']): r for r in records(folder / 'requests.jsonl')}
        commits = records(folder / 'actual_cut_commits.jsonl')
        if not commits:
            continue
        ready = {r['request_id']: r['time'] for r in records(folder / 'native_handoffs.jsonl')}
        admissions = {r['request_id']: r for r in records(folder / 'D_host_admission.jsonl')}
        decisions = records(folder / 'online_decisions.jsonl')
        decision_by_group = {r['generation']: r for r in decisions}
        cohort = collections.defaultdict(list)
        cuts = {}
        for index, commit in enumerate(commits):
            members = commit['requests']
            layers = commit['offloaded_layers']
            cuts.update({rid: layers for rid in members})
            decision = decision_by_group.get(commit.get('generation'))
            if decision is None and index < len(decisions):
                decision = decisions[index]
            decision_time = decision['time'] if decision else commit['time']
            key = str(len(members)) + ':' + str(layers)
            if len(members) <= cfg['P_slots'] or cfg.get('preplanned_source_groups'):
                value = 1000 * (max(ready[rid] for rid in members) - commit['time'])
                cells['P_source_ms'][key].append(value)
                origins['P_source_ms'][key].add(folder.name)
                cells['negotiation_ms'][key].append(1000 * (commit['time'] - decision_time))
                origins['negotiation_ms'][key].add(folder.name)
            for rid in members:
                row, admit = rows[rid], admissions[rid]
                components = dict(P_host_wait=1000 * (decision_time - row['arrival']),
                    decision_credit=1000 * (commit['time'] - decision_time),
                    P_submit_to_ready=1000 * (ready[rid] - commit['time']),
                    ready_to_D_admit=1000 * (admit['time'] - ready[rid]),
                    D_admit_to_first=1000 * (row['first_token'] - admit['time']))
                assert abs(sum(components.values()) - row['ttft_ms']) < 1e-5
                assert min(components.values()) > -.01
                decompositions.append(dict(name=folder.name, request_id=rid, layers=layers,
                    source_macro=len(members), TTFT_ms=row['ttft_ms'], components_ms=components))
                cohort[admit['cohort']].append(rid)
        prep_windows = []
        for members in cohort.values():
            layer_set = {cuts[rid] for rid in members}
            assert len(layer_set) == 1
            layers = next(iter(layer_set))
            key = str(len(members)) + ':' + str(layers)
            start = min(admissions[rid]['time'] for rid in members)
            end = max(rows[rid]['first_token'] for rid in members)
            cells['D_prepare_ms'][key].append(1000 * (end - start))
            origins['D_prepare_ms'][key].add(folder.name)
            actual_first = [rows[rid]['token_events'][1 if cfg.get('reuse_prefill_token') and layers == 0 else 0]['worker_time'] for rid in members]
            prep_windows.append((start, max(actual_first)))
        for stats_path in folder.glob('native_decode_*.json'):
            stats = json.loads(stats_path.read_text())
            window = json.loads((folder / 'window.json').read_text())
            for step in stats['host_step_spans']:
                if not window['start'] <= step['start'] <= window['end']:
                    continue
                if any(step['start'] < b and step['end'] > a for a, b in prep_windows):
                    continue
                key = str(step['outputs'])
                cells['decode_step_ms'][key].append(1000 * (step['end'] - step['start']))
                origins['decode_step_ms'][key].add(folder.name)
    table = {kind: {key: dict(summary(values), independent_source_starts=len(origins[kind][key]),
                             source_names=sorted(origins[kind][key])) for key, values in mapping.items()}
             for kind, mapping in cells.items()}
    write(ROOT / 'analysis' / ('host_component_decomposition_' + version + '.json'), dict(rows=decompositions,
        evidence='TRACE_ANALYSIS', scope='Exact telescoping host boundaries. These components sum to request TTFT. P submission includes queue, compute, state preparation and sampling. D preparation includes import, suffix, sampling and delivery. They are not exclusive GPU kernel durations.'))
    write(ROOT / 'configs' / ('empirical_host_cost_model_discovery_' + version + '.json'), dict(tables=table,
        scope='Discovery host service enclosures for the same8000-token model and runtime. Median is a planning estimate. Small correlated p95 samples are not WCET. Decode samples exclude the first preparation step. No clean kernel busy/idle inference.',
        evidence='MEASURED_HOST_ENCLOSURES'))
    print('HOST_COST_CELLS', {k: len(v) for k, v in table.items()}, flush=True)
    return table


def main():
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if not resources().processes():
            resources().restore(list(range(8)))
        fit()


if __name__ == '__main__':
    main()
