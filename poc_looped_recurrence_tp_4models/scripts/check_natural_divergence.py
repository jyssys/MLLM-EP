"""Bounded same-prefix diagnostic for the observed TP4 natural mismatch."""
import argparse
import gzip
import os
import subprocess
import sys
import time
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read


def collect(family, output):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,5,6,7'
    os.environ.update(POC4_MODEL='ouro', POC4_FAMILY=family, POC4_R='4')
    from vllm import LLM, SamplingParams
    reference = ROOT / 'results/raw' / f'ouro_{family}_R4_TP4_v1'
    kwargs = read(reference / 'engine_contract.json')
    kwargs['worker_extension_cls'] = 'poc_looped_recurrence_tp_4models.natural_probe.NaturalProbe'
    assert kwargs['enforce_eager'] is False
    output = Path(output)
    write(output / 'engine_contract.json', kwargs)
    llm = LLM(**kwargs)
    source = read(ROOT / 'results/diagnostics/ouro_native_R4_TP4_v4/natural_generation.json')['results'][0]
    assert source['request_id'] == 'math'
    prompt = source['prompt_ids']
    def generate(teacher):
        llm.collective_rpc('teacher', timeout=120, args=(teacher,))
        value = llm.generate([{'prompt_token_ids': prompt}], SamplingParams(
            temperature=0, max_tokens=256, ignore_eos=False), use_tqdm=False)[0]
        return dict(ids=list(value.outputs[0].token_ids), text=value.outputs[0].text,
            finish_reason=value.outputs[0].finish_reason)
    # Match the earlier natural shape. No retained-graph override is imported.
    natural = [generate(None) for _ in range(2)]
    write(output / 'natural_generation.json', dict(prompt_ids=prompt, results=natural))
    indices = [0, 111, 112, 113, 114, 115, 116, 127]
    llm.collective_rpc('natural_capture_start', timeout=120, args=(indices,))
    forced = generate(source['ids'])
    captured = llm.collective_rpc('natural_capture_finish', timeout=120)
    assert forced['ids'] == source['ids']
    with gzip.open(output / 'parity_natural_selected.json.gz', 'wt') as stream:
        json.dump(captured, stream)
    write(output / 'status.json', dict(success=True, family=family,
        selected_indices=indices, scope='Post-hoc natural discrepancy audit. Not clean timing or held-out confirmation.'))
    os._exit(0)


def summarize():
    import numpy as np
    root = ROOT / 'results/natural_recheck'
    records = []
    natural = {}
    for family in ['native', 'tied']:
        path = root / family
        assert read(path / 'status.json')['success']
        natural[family] = read(path / 'natural_generation.json')
        with gzip.open(path / 'parity_natural_selected.json.gz', 'rt') as stream:
            records.append(json.load(stream))
    assert natural['native']['prompt_ids'] == natural['tied']['prompt_ids']
    checks = []
    for rank, (a, b) in enumerate(zip(*records)):
        assert a['indices'] == b['indices'] and a['calls'] == b['calls']
        for index, step in enumerate(a['indices']):
            for field in ['logits', 'hidden']:
                x = np.asarray(a[field][index]); y = np.asarray(b[field][index])
                top = np.argsort(x)[-2:][::-1] if field == 'logits' else None
                checks.append(dict(rank=rank, step=step, field=field,
                    bitwise_equal=bool(np.array_equal(x, y)), max_abs=float(np.max(np.abs(x-y))),
                    native_top1=None if top is None else int(top[0]),
                    tied_top1=None if top is None else int(y.argmax()),
                    native_top2_margin=None if top is None else float(x[top[0]]-x[top[1]])))
    comparisons = []
    old = read(ROOT / 'results/diagnostics/ouro_native_R4_TP4_v4/natural_generation.json')['results'][0]['ids']
    for family, data in natural.items():
        ids = data['results'][0]['ids']
        comparisons.append(dict(family=family, repeat_parity=ids == data['results'][1]['ids'],
            matches_old_native=ids == old, output_tokens=len(ids)))
    a = natural['native']['results'][0]['ids']; b = natural['tied']['results'][0]['ids']
    write(ROOT / 'results/natural_discrepancy_audit.json', dict(
        same_policy_natural_token_parity=a == b,
        first_different_index=next((i for i,(x,y) in enumerate(zip(a,b)) if x != y), None),
        within_family=comparisons, selected_same_prefix_fields=checks,
        tolerance='Frozen bitwise policy. No relaxation.',
        limitation='Post-hoc calibration diagnostic. Selected steps and one prompt do not establish universal natural parity.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', choices=['native','tied']); parser.add_argument('--output')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize()
        return
    if args.collect:
        collect(args.collect, args.output)
        return
    deadline = time.monotonic() + 2400
    while True:
        path = ROOT / 'results/recheck_progress.json'
        try:
            ready = path.exists() and len(read(path)) == 4 and all(r['exit_code'] == 0 for r in read(path))
        except json.JSONDecodeError:
            ready = False
        if ready:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Do not overlap the natural audit with rechecks')
        time.sleep(15)
    for family in ['native','tied']:
        output = ROOT / 'results/natural_recheck' / family
        assert not output.exists(), 'Preserve all earlier evidence'
        command = [sys.executable, '-m', 'poc_looped_recurrence_tp_4models.scripts.gpu_session',
            '--gpus','4,5,6,7','--timeout','600','--output','results/natural_recheck/'+family,
            '--', PYTHON, '-m','poc_looped_recurrence_tp_4models.scripts.check_natural_divergence',
            '--collect',family,'--output',str(output)]
        subprocess.run(command, cwd=PROJECT, check=True)
    summarize()


if __name__ == '__main__':
    main()
