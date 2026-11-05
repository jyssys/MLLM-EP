"""Extend the pinned public corpus with explicitly disjoint held-out prompts."""
import hashlib
import random
import zipfile
from collections import Counter
from .common import *


def main():
    initialize()
    manifest = ROOT / 'state/heldout_data_provenance.json'
    if manifest.exists():
        print('Existing frozen held-out corpus retained')
        return
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    atomic = REPO / 'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'
    provenance = json.loads((atomic / 'configs/dataset_provenance.json').read_text())
    original = provenance['original']
    source = Path(original['cache_path'])
    assert digest(source) == original['source_SHA256']
    previous = json.loads((atomic / 'data/long_source_pool.json').read_text())
    excluded_ids = {row['prompt_id'] for row in previous}
    excluded_hashes = {row['source_SHA256'] for row in previous}
    prefixes = {model: {tuple(row['input_ids'][model][:128]) for row in previous}
                for model in ('ouro', 'huginn')}
    tokenizers = {model: AutoTokenizer.from_pretrained(str(MODEL_PATHS[model]),
                    trust_remote_code=True, local_files_only=True) for model in prefixes}
    lengths = dict(ouro=4096, huginn=2048)
    selected = {'ShareGPT': [], 'LongBench': []}
    scanned = 0

    def accept(messages, identifier, dataset):
        nonlocal scanned
        text = json.dumps(messages, sort_keys=True)
        sha = hashlib.sha256(text.encode()).hexdigest()
        if identifier in excluded_ids or sha in excluded_hashes or len(text) < 12000:
            return False
        scanned += 1
        ids = {}
        try:
            for model, tokenizer in tokenizers.items():
                value = tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_dict=False)
                value = list(value['input_ids'] if isinstance(value, dict) else value)
                if len(value) < lengths[model] or tuple(value[:128]) in prefixes[model]:
                    return False
                ids[model] = value[:lengths[model]]
        except (ValueError, TypeError):
            return False
        row = dict(prompt_id=identifier, source_dataset=dataset, source_SHA256=sha,
            input_ids=ids,
            source_type='LENGTH_MATCHED_CONTROLLED_PUBLIC_TEXT_PREFIX', truncation_explicit=True)
        selected['ShareGPT' if dataset == 'ShareGPT' else 'LongBench'].append(row)
        excluded_ids.add(identifier)
        excluded_hashes.add(sha)
        for model in ids:
            prefixes[model].add(tuple(ids[model][:128]))
        if sum(map(len, selected.values())) % 16 == 0:
            print('NEW_DISJOINT_PROMPTS', {key: len(value) for key, value in selected.items()}, flush=True)
        return True

    corpus = json.loads(source.read_text())
    random.Random(2026110506).shuffle(corpus)
    for item in corpus:
        messages = []
        for turn in item.get('conversations', []):
            role = {'human': 'user', 'gpt': 'assistant', 'system': 'system',
                    'user': 'user', 'assistant': 'assistant'}.get(turn.get('from'))
            if role and isinstance(turn.get('value'), str):
                messages.append(dict(role=role, content=turn['value']))
        users = [index for index, value in enumerate(messages) if value['role'] == 'user']
        if users:
            accept(messages[:users[-1] + 1], str(item.get('id', 'deep_sharegpt_' + str(scanned))), 'ShareGPT')
        if len(selected['ShareGPT']) >= 25:
            break
    assert len(selected['ShareGPT']) == 25, 'Insufficient independent ShareGPT texts'
    revision = provenance['supplement']['revision']
    archive = Path(hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset',
                                 revision=revision, local_files_only=True))
    assert digest(archive) == provenance['supplement']['source_SHA256']
    with zipfile.ZipFile(archive) as zipped:
        for task in ('qasper', 'multifieldqa_en', 'narrativeqa', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum'):
            names = [name for name in zipped.namelist() if name.endswith('/' + task + '.jsonl') or name == task + '.jsonl']
            if not names:
                continue
            values = [json.loads(line) for line in zipped.read(names[0]).decode().splitlines()]
            random.Random(2026110507).shuffle(values)
            for index, value in enumerate(values):
                content = 'Document:\n' + value['context'] + '\nQuestion:\n' + value.get('input', '')
                accept([dict(role='user', content=content)],
                       f'longbench_{task}_{value.get("_id", index)}', 'LongBench/' + task)
                if len(selected['LongBench']) >= 135:
                    break
            if len(selected['LongBench']) >= 135:
                break
    assert len(selected['LongBench']) == 135, 'Insufficient independent LongBench texts'
    pools = dict(calibration=selected['ShareGPT'][:5] + selected['LongBench'][:27],
                 heldout=selected['ShareGPT'][5:] + selected['LongBench'][27:])
    for split, rows in pools.items():
        random.Random(2026110508).shuffle(rows)
        for model, length in lengths.items():
            values = [dict(row, input_ids={model: row['input_ids'][model]}, length=length) for row in rows]
            assert len({tuple(row['input_ids'][model]) for row in values}) == len(values)
            write(ROOT / 'data' / f'{split}_{model}_L{length}.json', values)
    write(manifest, dict(time=time.time(), source_provenance=provenance,
        excluded_previous_sources=len(previous), selection_seeds=[2026110506, 2026110507, 2026110508],
        pools={split: dict(count=len(rows), unique_IDs=len({row['prompt_id'] for row in rows}),
                          sources=dict(Counter(row['source_dataset'] for row in rows)),
                          IDs=[row['prompt_id'] for row in rows]) for split, rows in pools.items()},
        overlap_previous_IDs=0, overlap_previous_source_hashes=0, overlap_previous_first128_token_prefixes=0,
        overlap_calibration_heldout_IDs=0, tokenizer_models={model: str(MODEL_PATHS[model]) for model in lengths},
        scope='Pinned ShareGPT and LongBench extension. Explicit input-length prefixes and a 12000-character prefilter. Not natural EOS, Azure traffic, or a population-random sample.'))
    print('FROZEN_HELDOUT_128_AND_CALIBRATION_32', flush=True)


if __name__ == '__main__':
    main()
