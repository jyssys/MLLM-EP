"""Freeze disjoint baseline requests before the new robustness runs."""
import argparse
import datetime as dt
import random

from poc_ouro_parallelism_deep_discovery.contract import MODELS, PROJECT, ROOT, read, sha, write


def freeze(output):
    if output.exists():
        raise FileExistsError(output)
    source = PROJECT / 'poc_decode_first_depth_aware_dp/artifacts/quality_protocol_v3_pilot32.json'
    old_path = ROOT / 'artifacts/probe_protocol.json'
    prior_path = PROJECT / 'poc_depth_cdb_factorial/artifacts/workload_frozen.json'
    pool, old, prior = read(source), read(old_path), read(prior_path)
    excluded = {row['id'] for row in old['rows']}
    excluded.update(row['id'] for row in pool['rows'] if row['split'] == 'pilot')
    # Read request metadata only. Do not inspect protected experiment results.
    excluded.update(row['id'] for key in ('quality', 'speed') for row in prior[key])
    selected = []
    for domain in ('gsm8k', 'math500'):
        eligible = [row for row in pool['rows'] if row['dataset'] == domain and row['id'] not in excluded]
        random.Random(202610191721).shuffle(eligible)
        if len(eligible) < 32:
            raise ValueError('Too few disjoint baseline requests')
        selected.extend(dict(row, campaign_partition='baseline_extension') for row in eligible[:32])
    random.Random(202610191722).shuffle(selected)
    ids = [row['id'] for row in selected]
    assert len(ids) == len(set(ids)) == 64 and not set(ids) & excluded
    protocol = dict(status='FROZEN_BEFORE_NEW_OUTCOMES', frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Baseline robustness and original-gate work budget. Not candidate confirmation or architecture speedup.',
        source=str(source.relative_to(PROJECT)), source_sha256=sha(source),
        existing_protocol_sha256=sha(old_path), protected_metadata_sha256=sha(prior_path),
        protected_results_inspected=False, selection_uses_outcomes=False,
        exclusions='All original80 campaign IDs, earlier pilot IDs, and protected factorial request IDs.',
        original_held_out='UNTOUCHED. No original held-out request enters this extension.',
        rows=selected, N=64, per_domain=32, batch=1, quality_cap=8192, temperature=0, EOS=0,
        graphs=True, precision='BF16', new_policy='NONE', independent_gpu_per_arm=True,
        modes=['fixed', 'adaptive'], sizes=list(MODELS),
        fixed=dict(min_R=4, max_R=4, q=1.0, prefill='full R4'),
        adaptive=dict(min_R=2, max_R=4, q=0.5, prefill='full R4', gate='original checkpoint', KV='last_exited'),
        pins={size: dict(id=value['id'], revision=value['revision']) for size, value in MODELS.items()},
        interpretation='Request-level confidence intervals only. No token-level population inflation. Adaptive-vs-fixed is operating-point context.',
        stop_rule='Preserve failures. A baseline failure does not become candidate quality loss. No tolerance or token-cap change after outcomes.')
    write(output, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = freeze(ROOT / args.output)
    print(dict(N=value['N'], per_domain=value['per_domain'], status=value['status']))


if __name__ == '__main__':
    main()
