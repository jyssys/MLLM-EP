"""Secondary native-stage latency analysis from actual engine completions.

This does not replace the complete injection-window peak metric or select a
new LP configuration. It separates the first producer/vision iteration from
later prefill stages to describe the granularity sweep. Host completion stamps
are never labeled GPU collective start/ready clocks or CWT.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from poc_native_lp_ep4.contract import atomic_json, read_json
from poc_native_lp_ep4.scripts.analyze import victim_itls
from poc_native_lp_ep4.scripts.evidence import estimate


def align_completion(completion_ns, token_ends, tolerance_ms=5):
    """Match a scheduler completion to its emitted DP0 resident token.

    Assertion is fail-closed: an absent or distant stamp is not interpolated
    into a fictitious serving iteration.
    """
    token_ends = np.asarray(token_ends, dtype=np.int64)
    if not len(token_ends):
        raise ValueError('no resident token completion to align')
    index = int(np.argmin(np.abs(token_ends - int(completion_ns))))
    error_ms = abs(int(token_ends[index]) - int(completion_ns)) / 1e6
    if error_ms > tolerance_ms:
        raise ValueError(f'stage/token completion mismatch: {error_ms:.3f} ms')
    return index, error_ms


def stage_rows(run):
    manifest = read_json(run / 'manifest.json')
    if manifest['trace']:
        raise ValueError('stage latency headline uses clean runs only')
    decoded = manifest['decode_tokens']
    workers = [read_json(run / f'worker.dp{dp}.json')['waves'] for dp in (0, 1)]
    markers = defaultdict(list)
    for path in run.glob('completion*.jsonl'):
        for line in path.open():
            event = json.loads(line)
            if event.get('kind') == 'prefill_stage_completion':
                markers[event['wave']].append(event)
    rows = []
    for wave in workers[0]:
        if wave['repeat'] < 0 or wave['arm'] == 'DECODE_ONLY':
            continue
        partner = next(w for w in workers[1] if w['wave'] == wave['wave'])
        local, ends = victim_itls(wave, decoded, True)
        peer = victim_itls(partner, decoded)
        events = sorted(markers[wave['wave']], key=lambda e: e['completion_ns'])
        if not events:
            raise ValueError(f'missing native completion markers: {run}/{wave["wave"]}')
        if wave.get('baseline_wave') is not None:
            baseline = next(w for w in workers[0] if w['wave'] == wave['baseline_wave'])
            bp = next(w for w in workers[1] if w['wave'] == baseline['wave'])
            base = victim_itls(baseline, decoded) + victim_itls(bp, decoded)
        else:
            base = None
        for ordinal, event in enumerate(events):
            index, error = align_completion(event['completion_ns'], ends[0])
            values = [float(a[index]) for a in local + peer]
            excess = None if base is None else [values[i] - float(b[index]) for i, b in enumerate(base)]
            rows.append({'run': str(run), 'wave': wave['wave'], 'repeat': wave['repeat'],
                'arm': wave['arm'], 'modality': wave['modality'],
                'ordinal': ordinal, 'phase': 'FIRST_PRODUCER_ITERATION' if ordinal == 0 else 'LATER_PREFILL_STAGE',
                'resident_generated_token_ordinal': index + 2, 'stage_markers': event['markers'],
                'dp0_completion_alignment_error_ms': error,
                'victim_mean_itl_ms': float(np.mean(values)), 'victim_max_itl_ms': max(values),
                'victim_max_vsel_ms': None if excess is None else max(excess),
                'scope': 'SECONDARY_ACTUAL_ENGINE_COMPLETION; SAME_DECODE_POSITION_ACROSS_DP; NOT_GPU_CWT'})
    return rows


def summarize_stages(rows):
    # Reduce every wave to one sample before bootstrapping. Do not count its
    # layers, stages or eight residents as independent serving replicates.
    waves = defaultdict(list)
    for row in rows:
        waves[(row['run'], row['wave'], row['phase'])].append(row)
    per_wave = []
    for (run, wave, phase), items in waves.items():
        first = items[0]
        per_wave.append({k: first[k] for k in ('run', 'wave', 'phase', 'arm', 'modality', 'repeat')} | {
            'stages': len(items),
            'mean_stage_itl_ms': float(np.mean([r['victim_mean_itl_ms'] for r in items])),
            'peak_stage_itl_ms': max(r['victim_max_itl_ms'] for r in items),
            'peak_stage_vsel_ms': None if all(r['victim_max_vsel_ms'] is None for r in items) else
                max(r['victim_max_vsel_ms'] for r in items if r['victim_max_vsel_ms'] is not None)})
    groups = defaultdict(list)
    for row in per_wave:
        groups[(row['arm'], row['modality'], row['phase'])].append(row)
    summary = []
    for (arm, modality, phase), items in groups.items():
        summary.append({'arm': arm, 'modality': modality, 'phase': phase, 'n_waves': len(items),
            **{key: estimate([r[key] for r in items if r[key] is not None]) for key in
               ('mean_stage_itl_ms', 'peak_stage_itl_ms', 'peak_stage_vsel_ms')}})
    return per_wave, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = sum((stage_rows(run) for run in args.runs), [])
    waves, summary = summarize_stages(rows)
    atomic_json(args.output, {'scope': 'SECONDARY_NOT_SELECTION; PRIMARY_WHOLE_WINDOW_UNCHANGED',
        'rows': rows, 'wave_summaries': waves, 'summary': summary,
        'limits': ['DP0 scheduler completion is aligned to actual resident engine completion.',
                   'Peer ITL uses the same forced token index, not a cross-GPU clock subtraction.',
                   'No serving samples, GPU timestamps or isolated network service are simulated.']})


if __name__ == '__main__':
    main()
