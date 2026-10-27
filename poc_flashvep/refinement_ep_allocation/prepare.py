"""Freeze train-only selection splits and complete public test manifests."""
import json
import random
import subprocess
from pathlib import Path
from .common import ROOT, REPO, MODEL, REVISION, REFERENCE, DINFER, setup, write, digest

def main():
    setup()
    import pyarrow.ipc as ipc
    import yaml as yaml_module
    cache = Path('/home/esjung/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866')
    yaml = DINFER/'evaluations/tasks/gsm8k/gsm8k-llada-mini.yaml'
    train = ipc.open_stream(str(cache/'gsm8k-train.arrow')).read_all().to_pylist()
    test = ipc.open_stream(str(cache/'gsm8k-test.arrow')).read_all().to_pylist()
    examples = yaml_module.safe_load(yaml.read_text())['fewshot_config']['samples'][:4]
    prefix = '\n\n'.join(f"Question: {r['question']}\nLet's think step by step\nAnswer: {r['answer']}" for r in examples)
    indices = list(range(len(train)))
    random.Random(20261027).shuffle(indices)
    manifests = {}
    for split, source, ids in [('calibration', train, indices[:256]),
                               ('validation', train, indices[256:384]),
                               ('gsm8k_test', test, list(range(len(test))))]:
        data = []
        for i in ids:
            r = source[i]
            data.append(dict(sample_id=f'gsm8k:{split}:{i}', id=i, task='gsm8k', split=split,
                             question=r['question'],
                             raw_prompt=prefix+f"\n\nQuestion: {r['question']}\nLet's think step by step\nAnswer:",
                             # Policy selection receives no train labels.
                             answer=r['answer'] if split == 'gsm8k_test' else None,
                             gen_length_cap=2048))
        manifests[split] = data
    old = Path('/home/esjung/MLLM-EP-mini-repro/artifacts/reproduction/humaneval_samples.jsonl')
    he = [json.loads(line) for line in old.read_text().splitlines() if line.strip()]
    assert len(he) == 164 and len({r['sample_id'] for r in he}) == 164
    manifests['humaneval_test'] = [dict(sample_id=f"humaneval:test:{r['sample_id']}",
        id=r['sample_id'], task='humaneval', split='humaneval_test',
        question=r['original_prompt'], raw_prompt=r['raw_prompt'], answer=None,
        test=r['test'], entry_point=r['entry_point'], gen_length_cap=1024) for r in he]
    manifests['sanity'] = [dict(sample_id=f'sanity:{i}', id=i, task='sanity', split='sanity',
        raw_prompt=q, question=q, answer=None, gen_length_cap=64)
        for i, q in enumerate(('What is 7 + 8? Give the answer.', 'What is 42 - 17? Give the answer.'))]
    hashes = {}
    for name, data in manifests.items():
        path = ROOT/'manifests'/f'{name}.jsonl'
        payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data)
        if path.exists():
            assert path.read_text() == payload, f'Frozen manifest changed: {path}'
        else:
            path.write_text(payload)
        hashes[name] = dict(n=len(data), sha256=digest(path), path=str(path))
    assert set(indices[:256]).isdisjoint(indices[256:384])
    write(ROOT/'manifests/validation_truth.json', {
        f'gsm8k:validation:{i}':train[i]['answer'] for i in indices[256:384]})
    source_files = [MODEL/'config.json', MODEL/'tokenizer_config.json', MODEL/'modeling_llada2_moe.py',
                    REFERENCE/'run_reference.py', REFERENCE/'score_humaneval.py', yaml,
                    DINFER/'python/dinfer/model/modeling_llada2_moe_sglang.py']
    write(ROOT/'run_config.json', dict(model='inclusionAI/LLaDA2.0-mini', revision=REVISION,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        protocol='mini_reference_legacy_matched_v1', temperature=0, threshold=.95,
        block_length=32, steps_per_block_max=32, sample_seed_base=20260915,
        split_seed=20261027, manifests=hashes,
        source_hashes={str(p):digest(p) for p in source_files},
        official_protocol_identical=False,
        caveat='HF temperature=0 uses multinomial. Native decoder uses greedy. Compare them as distinct protocols.'))
    print(json.dumps(hashes, indent=2))

if __name__ == '__main__':
    main()
