"""Audit sampled slot pressure without treating request phases as GPU activity."""
import bisect
from collections import Counter
from .common import *


def classify(snapshot):
    compute = any(not any(word in row['phase'].lower()
        for word in ('wait', 'receiv', 'transfer', 'finish')) for row in snapshot['requests'])
    submitted = bool(snapshot['all_submitted_unretired_ids'])
    transfer_only = (not compute and not submitted
        and any('TRANSFERRING' in row['phase'] for row in snapshot['requests']))
    return compute, submitted, transfer_only


def one(name):
    folder = ROOT / 'raw' / name
    window = json.loads((folder / 'window.json').read_text())
    left, right = window['start'], window['last_arrival']
    snapshots = []
    for path in folder.glob('memory_progress_*.jsonl'):
        values = [json.loads(line) for line in path.read_text().splitlines()]
        snapshots.extend(row for row in values if row['role'] == 'prefill')
    snapshots.sort(key=lambda row: row['time'])
    times = [row['time'] for row in snapshots]
    counts = Counter()
    for line in (folder / 'admission_reasons.jsonl').read_text().splitlines():
        row = json.loads(line)
        if not left <= row['time'] <= right:
            continue
        counts['admission_samples'] += 1
        if not any(waiter['reasons'] == ['PREFILL_NO_SLOT'] for waiter in row['waiting']):
            continue
        counts['D_clear_P_slot_samples'] += 1
        index = bisect.bisect_right(times, row['time']) - 1
        if index < 0 or row['time'] - snapshots[index]['time'] > 1.:
            continue
        counts['fresh_owner_samples'] += 1
        compute, submitted, transfer_only = classify(snapshots[index])
        label = ('COMPUTE_PHASE' if compute else 'NO_COMPUTE_PHASE') + (
                 '_UNRETIRED' if submitted else '_NO_UNRETIRED')
        counts[label] += 1
        counts['source_transfer_only_samples'] += int(transfer_only)
    return dict(name=name, observation_start=left, observation_end=right, counts=dict(counts),
        source_SHA256={str(path.relative_to(ROOT)): digest(path) for path in
            [folder / 'window.json', folder / 'admission_reasons.jsonl',
             *folder.glob('memory_progress_*.jsonl')]},
        scope='Admission-sample counts, not elapsed-time fractions. Join the latest preceding P-owner snapshot only within one second. '
            'PREFILL_NO_SLOT must be the only refusal reason. Unretired submissions remain separate from host request phases. '
            'Neither a host phase nor a submitted ticket proves actual GPU activity. No GPU idle or causal DMA cost is inferred.')


def main():
    initialize()
    names = ['f_ouro_heavy_native_start1_128', 'f_ouro_heavy_loop3_start1_128',
             'f_ouro_heavy_token3072_start1_128', 'n_ouro_heavy_token3840_start1_128']
    results = [one(name) for name in names]
    write_derived(ROOT / 'state/source_slot_sample_audit.json', dict(results=results,
        verdict='NO_DOMINANT_TRANSFER_ONLY_STATE_IN_RECORDED_SAMPLES',
        limitation='Sparse owner snapshots cannot exclude short transfer stalls between samples. This is not a causal throughput experiment.'))


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('source_slot_sample_audit'):
        main()
