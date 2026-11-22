"""Test pipeline saturation separately from the earlier serial-rate reference."""
import json
from fractions import Fraction
import numpy as np
from .common import ROOT, write, digest
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .next_jobs import plan
from .service_accounting import analyze


def exact_subset(bucket):
    pool = json.loads((ROOT / 'data' / f'mechanism_input{bucket}_discovery.json').read_text())
    order = np.random.default_rng(2026112461 + bucket).permutation(len(pool)).tolist()
    states = {(0, 0): ()}
    for i in order:
        for (count, remainder), selected in sorted(list(states.items()), reverse=True):
            if count == 32:
                continue
            key = count + 1, (remainder + pool[i]['input_tokens']) % 1152
            if key not in states:
                states[key] = selected + (i,)
        if (32, 0) in states:
            break
    feasible = [count for count in range(16, 33) if (count, 0) in states]
    assert feasible, 'Do not pad or duplicate prompts to satisfy exact work'
    count = max(feasible)
    selected = [pool[i] for i in states[count, 0]]
    assert len({r['prompt_id'] for r in selected}) == count
    name = f'pipeline_input{bucket}_discovery_exact{count}_v1.json'
    path = ROOT / 'data' / name
    if path.exists():
        assert json.loads(path.read_text()) == selected
    else:
        write(path, selected)
    return name, selected


def main():
    capsule, sha = freeze('proper_pipeline_load_discovery_v1_20261121')
    datasets = {b: exact_subset(b) for b in (0, 3)}
    refs = {}
    for bucket, output in ((0, 8), (0, 16), (0, 64), (3, 8)):
        name = f'CapacityC0_I{bucket}_O{output}_Serial16_v1'
        cost = analyze(name)['means']
        refs[bucket, output] = dict(P_mean_ms=cost['P_submit_to_ready_ms'],
            D_whole_request_mean_ms=cost['D_submit_to_finish_ms'],
            estimated_pipeline_rate=1000 / max(cost['P_submit_to_ready_ms'], cost['D_submit_to_finish_ms']),
            source=name)
    cells = []
    for bucket in (0, 3):
        for load in (.75, 1., 1.15):
            cells.append(dict(input_bucket=bucket, output=8, QPS=refs[bucket, 8]['estimated_pipeline_rate'] * load,
                reference_relative_load=load, family='PIPELINE_LOAD_DISCOVERY'))
    # Same-q controls isolate output residency. O64 also retains a near-own-rate control.
    same_q = refs[0, 8]['estimated_pipeline_rate']
    for output in (16, 64):
        cells.append(dict(input_bucket=0, output=output, QPS=same_q,
            reference_relative_load=same_q / refs[0, output]['estimated_pipeline_rate'], family='SAME_Q_OUTPUT_COUNTERFACTUAL'))
    cells.append(dict(input_bucket=0, output=64, QPS=refs[0, 64]['estimated_pipeline_rate'],
        reference_relative_load=1., family='LONG_OUTPUT_OWN_RATE_CONTROL'))
    lanes = [[] for _ in range(4)]
    records = []
    for index, cell in enumerate(cells):
        lane = index % 4
        first = 2 * lane
        bucket, output = cell['input_bucket'], cell['output']
        dataset, data = datasets[bucket]
        lengths = [r['input_tokens'] for r in data]
        actions = [('Native', 0)] + [(axis, m) for m in (2, 4, 8, 12) for axis in ('Layer', 'Token')] + [('Hybrid', 4)]
        jobs = []
        for axis, cut in actions:
            extra = {}
            if axis != 'Native':
                extra['qualification_gate'] = 'C0_full_state_q4_gate.json'
            if axis == 'Token':
                suffix, work = matched_suffixes(lengths, cut)
                assert work['exact_match']
                extra.update(suffix_by_request={'q' + str(i): s for i, s in enumerate(suffix)}, aggregate_work_audit=work)
            if axis == 'Hybrid':
                suffix, work = matched_suffixes(lengths, Fraction(9, 2))
                assert work['exact_match']
                extra.update(suffix_by_request={'q' + str(i): s for i, s in enumerate(suffix)},
                    hybrid_suffix_fraction=.125, partition_aligned_chunks=True,
                    qualification_gate='C0_family_q7_gate.json',
                    target_D_rows=sum(lengths) * 8, target_P_rows=sum(lengths) * 28)
            name = create(f'PipelineLoad_I{bucket}_O{output}_C{index}_{axis}{cut}_N{len(data)}_v1', axis, dataset,
                cut=cut, pair=(first, first + 1), qps=cell['QPS'], seed=2026112461,
                output=output, fixed=True, controlled=True, discovery_only=True,
                source_capsule=capsule, source_capsule_SHA256=sha,
                pipeline_load_reference=refs[bucket, 8],
                scope='C0 actual1/1/1,credit4. New load-regime supplement. '
                    'Earlier discovery normalized by serial P+D service and often stayed below pipeline saturation. '
                    'This estimate uses max(P mean,D whole-request mean),not their sum. '
                    'It is a reference estimate,not measured sustainable capacity. '
                    'All methods retain the same exact original prompts,arrival trace and fixed output. '
                    'No old result or SLO changes. Confirm any winner on new requests and seeds.', **extra)
            jobs.append(name)
        np.random.default_rng(2026112461 + index).shuffle(jobs)
        lanes[lane].extend(jobs)
        records.append(dict(**cell, lane=lane, jobs=jobs, request_count=len(data), dataset=dataset,
            dataset_SHA256=digest(ROOT / 'data' / dataset)))
    for lane, jobs in enumerate(lanes):
        plan(f'proper_pipeline_load_discovery_lane{lane}_v1', jobs)
    write(ROOT / 'configs/proper_pipeline_load_discovery_frozen_v1.json', dict(cells=records,
        source_capsule_SHA256=sha, references=[dict(input_bucket=b, output=o, **v) for (b, o), v in refs.items()],
        evidence='FROZEN_DISCRIMINATING_LOAD_DESIGN',
        hypotheses='At equal transformer rows,the winner can change when the bottleneck changes from P service '
            'to D continuation plus decode residency. Serial P+D is not pipeline capacity. '
            'Test short-input Layer candidates and long-input Token controls. Keep severe-output overload controls.',
        qualifier='One discovery start. Select the largest exact-work subset of16 through32 original discovery prompts. '
            'The long-input pool cannot supply an exact32-request subset. Preserve the count difference across input strata. '
            'Every axis within one cell uses exactly the same requests. No duplicate or padded prompt. '
            'Queue growth and achieved rate must classify each measured trace. '
            'The original64-cell light-load confirmation remains valid in its original scope. '
            'Do not relabel these new points as an original preregistered matrix or a natural400 result.'))
    print('Frozen', sum(len(v) for v in lanes), 'pipeline-load discovery starts', flush=True)


if __name__ == '__main__':
    main()
