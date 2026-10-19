"""Build result panels without joining incompatible quality and timing lanes."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_panels(value):
    rows = [dict(row) for row in value['summaries'] if row['status'] == 'AVAILABLE']
    lookup = {(r['size'], r['mode'], r['C'], r['context']): r for r in rows}
    require(len(lookup) == len(rows), 'Native operating point repeats')
    result = []
    for row in rows:
        reference = lookup[(row['size'], 'TP1', row['C'], row['context'])]
        is_api = row['mode'] in ('TP1', 'TP2', 'TP4') and row['boundary'] == reference['boundary']
        result.append(dict(size=row['size'], depth_mode='fixed R4', method=row['mode'],
            GPUs=row['reserved_GPUs'], C=row['C'], context=row['context'], output_tokens_per_request=128,
            median_wall_ms=row['median_wall_ms'], repetitions=row['repetitions'], boundary=row['boundary'],
            speedup_vs_native_TP1=reference['median_wall_ms']/row['median_wall_ms'] if is_api else None,
            speedup_vs_strongest_same_P=None, output_TPS=row['median_output_TPS'], quality='NOT_RUN',
            evidence_class='LIVE_E2E', scope='Controlled constant-token hardware workload, not natural task performance.',
            strongest_same_P_rule='NOT_ESTABLISHED across different API and parent timing endpoints.',
            uncertainty='Five repeats within one sequential job and frozen shapes. No randomized architecture CI.'))
    return result


def quality_panels(value):
    result = []
    for row in value['summaries']:
        result.append(dict(cohort='Original calibration:8/domain', size=row['size'], mode=row['mode'],
            runtime=row['runtime'], GPUs=row['TP'], dataset=row['dataset'], N=row['N'], correct=row['correct'],
            accuracy_percent=row['accuracy_percent'], accuracy_CI_percent=row['accuracy_CI_percent'],
            parsing_failures=row['parsing_failures'], truncated=row['truncated'],
            raw_latency='NOT_RUN in this quality panel', speedup_vs_native_1GPU='NOT_RUN',
            speedup_vs_strongest_same_P='NOT_RUN', evidence_class='QUALITY_ONLY'))
    return result


def extension_panels(value):
    require(value['evidence_class'] == 'QUALITY_ONLY', 'Extension is not a quality lane')
    available = {(row['size'],row['mode']) for row in value.get('jobs',[]) if row['status']=='AVAILABLE'}
    result = []
    for row in value['summaries']:
        require(row['N'] == 32, 'Extension domain is incomplete')
        require((row['size'],row['mode']) in available, 'Extension summary has no complete admitted arm')
        records=[r for r in value.get('records',[]) if (r['size'],r['mode'])==(row['size'],row['mode'])]
        require(len(records)==len({r['id'] for r in records})==64, 'Extension arm records are incomplete or repeated')
        require(sum(r['dataset']==row['dataset'] for r in records)==32, 'Extension summary domain differs from raw admission')
        result.append(dict(cohort='Disjoint baseline extension:32/domain', size=row['size'], mode=row['mode'],
            runtime='staged_reference, graph-enabled B1', GPUs=1, dataset=row['dataset'], N=row['N'],
            correct=row['correct'], accuracy_percent=row['accuracy_percent'],
            accuracy_CI_percent=row['accuracy_CI_percent'], parsing_failures=row['parsing_failures'],
            truncated=row['truncated'], raw_latency='NOT_RUN in this quality panel',
            speedup_vs_native_1GPU='NOT_RUN', speedup_vs_strongest_same_P='NOT_RUN', evidence_class='QUALITY_ONLY'))
    return result


def render_quality(rows):
    lines = ['| Model / mode | Runtime | GPUs | Domain | Correct/N | Accuracy,95% CI | Parse failures | Truncated |',
             '|---|---|---:|---|---:|---|---:|---:|']
    for r in rows:
        lo, hi = r['accuracy_CI_percent']
        lines.append(f"| Base{r['size']} / {r['mode']} | {r['runtime']} | {r['GPUs']} | {r['dataset']} | "
                     f"{r['correct']}/{r['N']} | {r['accuracy_percent']:.3f}%,[{lo:.3f},{hi:.3f}]% | "
                     f"{r['parsing_failures']} | {r['truncated']} |")
    return '\n'.join(lines)


def render(value):
    lines = ['# Current result panels: specification Tables C and D', '',
        'This is an active-campaign synthesis. It is not the final campaign verdict.',
        'The tables preserve separate evidence lanes. They do not certify the campaign duration.', '',
        '## Table C1: original natural-generation calibration', '',
        'These are actual greedy natural outputs from the original8 requests per domain.',
        'Each runtime and TP setting remains a separate operating point.',
        'No row supplies candidate quality or a matched architecture speedup.',
        'Request-level95% bootstrap intervals describe this small calibration sample.',
        'They do not establish1pp or2pp noninferiority. The original held-out remains untouched.', '',
        render_quality(value['original_quality']), '',
        'The table retains backend/TP output differences and length truncation.',
        'It does not promote partial teacher-forced agreement into natural-output parity.', '',
        '## Table C2: disjoint baseline extension', '',
        'The frozen extension excludes the original calibration and held-out IDs.',
        'It uses32 new requests per domain and the unchanged original gate.',
        'Only complete, independently audited arms enter this panel.',
        'The extension is baseline evidence. It is not candidate held-out confirmation.', '']
    lines.append(render_quality(value['extension_quality']) if value['extension_quality'] else
                 'No complete extension arm is admitted at this checkpoint.')
    lines += ['', 'The extension does not pool its requests with the earlier8/domain cohort.',
        'Adaptive-minus-fixed comparisons describe different original operating points, not a new mechanism.', '',
        '## Table C3: controlled native hardware workload', '',
        'These native calls execute real pretrained weights and real physical TP collectives.',
        'Each request emits128 controlled constant tokens. EOS does not shorten this hardware workload.',
        'Each cell reports the median of five trials. This is not task accuracy or natural reasoning latency.',
        'TP rows use the native API wall. DP rows use the parent release-to-completion wall.',
        'The two endpoints are not interchangeable. DP speedups against a TP endpoint remain NOT_RUN.',
        'TP1 ratios divide matching cohort/context medians. Sequential jobs do not establish randomized paired uncertainty.',
        'No candidate earns a same-P speedup from these baseline rows.', '',
        '| Model | Method | GPUs | C | Prefix | Median wall,ms | Output TPS | TP1 ratio |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in value['native_hardware']:
        ratio = f"{r['speedup_vs_native_TP1']:.3f}×" if r['speedup_vs_native_TP1'] is not None else 'NOT_RUN'
        lines.append(f"| Base{r['size']} | {r['method']} | {r['GPUs']} | {r['C']} | {r['context']} | "
                     f"{r['median_wall_ms']:.3f} | {r['output_TPS']:.3f} | {ratio} |")
    lines += ['', 'All four-GPU configurations reserve208GiB aggregate KV,52GiB per physical rank.',
        'TP4, TP2×DP2, and DP4 differ in weight multiplicity and KV ownership.',
        'TP1 and TP2 use fewer physical ranks. These reservations do not measure HBM reads.',
        'The separate32K native protocol reserves52GiB aggregate KV. Its rows must not be merged here.', '',
        '## Table C4: candidate performance and quality', '',
        '| Candidate set | Both Base sizes | Fixed / adaptive | GSM8K / MATH | Live architecture latency | Both speedups |',
        '|---|---|---|---|---|---|',
        '| C01–C19 | Target scope only | Separate unimplemented paths | NOT_RUN | NOT_RUN | NOT_RUN |', '',
        'Component probes, numerical diagnostics, and source results do not fill these cells.', '',
        '## Table D: charged headroom and missing measurements', '',
        'UNRESOLVED means that no valid complete critical-path value exists.',
        'A scoped premise failure is not a universal physical impossibility claim.', '',
        '| Candidate | Baseline exposed time | Removed/overlapped cost | Added compute | Added communication | Added synchronization/state | Critical-path estimate | Live validation/error |',
        '|---|---|---|---|---|---|---|---|']
    for r in value['headroom']:
        lines.append('| ' + ' | '.join(str(r[key]) for key in
            ('candidate','baseline_exposed','removed','compute','communication','state','estimate','validation')) + ' |')
    lines += ['', '## Reproduction and scope', '',
        'The structured summary records all input hashes and full timing endpoints.',
        'The headroom supplement retains a source path for every candidate row.',
        'The frozen card labels remain unchanged. Later scope corrections control the interpretation.',
        'The CDB observer failed validity guards. It supplies no removable-time share or Amdahl percentage.',
        'No actual HBM-counter result exists. No P8 hardware measurement exists.',
        'The primary-claim re-audit controls source scope. Source-reported speedups remain separate from our measurements.', '']
    return '\n'.join(lines)


def build(root, hybrid_path, headroom_path, extension_path=None):
    root = Path(root).resolve()
    hybrid_path, headroom_path = Path(hybrid_path).resolve(), Path(headroom_path).resolve()
    if extension_path:
        extension_path=Path(extension_path).resolve()
    original = root/'artifacts/quality_checkpoint9.json'
    inputs = [original, hybrid_path, headroom_path]
    hardware = read(hybrid_path)
    headroom = read(headroom_path)
    require([r['candidate'] for r in headroom['rows']] == [f'C{i:02}' for i in range(1,20)], 'Headroom IDs differ')
    result = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_quality=quality_panels(read(original)), native_hardware=native_panels(hardware),
        extension_quality=[], headroom=headroom['rows'], candidate_LIVE_E2E='NOT_RUN',
        candidate_free_running_quality='NOT_RUN', original_held_out='UNTOUCHED',
        boundary_rule=hardware['boundary_rule'], memory_rule=hardware['memory_rule'],
        source_scope='See reports/primary_claim_reaudit_v1.md. No source result becomes our GPU evidence.')
    if extension_path:
        inputs.append(extension_path)
        result['extension_quality'] = extension_panels(read(extension_path))
    result['input_sha256'] = {str(p.relative_to(root)):sha(p) for p in inputs}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extension', type=Path)
    args=parser.parse_args()
    require(not args.output.exists(), 'Use a new output directory')
    value=build(ROOT, ROOT/'analysis/native_hybrid_cohorts_complete_v3/summary.json',
                ROOT/'artifacts/current_headroom_decomposition_v1.json', args.extension)
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (args.output/'tables_cd.md').write_text(render(value))
    for key in ('original_quality','extension_quality','native_hardware','headroom'):
        rows=value[key]
        if rows:
            with (args.output/f'{key}.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(dict(native_cells=len(value['native_hardware']),
        original_quality_cells=len(value['original_quality']), extension_cells=len(value['extension_quality']),
        headroom_rows=len(value['headroom']))))


if __name__=='__main__':
    main()
