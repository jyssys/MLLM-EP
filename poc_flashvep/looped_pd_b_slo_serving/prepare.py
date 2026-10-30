"""Pin real ShareGPT text and disjoint workload subsets before B serving."""
import hashlib
import json
import random
import urllib.request

from .common import ROOT, MODELS, setup, write


def main():
    setup()
    existing = ROOT / 'configs/dataset_provenance.json'
    if existing.exists():
        lock = ROOT / 'configs/dataset_lock.json'
        if lock.exists():
            for name, expected in json.loads(lock.read_text()).items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('A frozen dataset manifest changed: ' + name)
        print('REUSE_EXISTING_DATASET_PROVENANCE', flush=True)
        return
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    repo = 'anon8231489123/ShareGPT_Vicuna_unfiltered'
    filename = 'ShareGPT_V3_unfiltered_cleaned_split.json'
    info = HfApi().dataset_info(repo)
    source = hf_hub_download(repo, filename, repo_type='dataset', revision=info.sha)
    with open(source) as stream:
        conversations = json.load(stream)
    tokenizers = {model: AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
                  for model, path in MODELS.items() if model in ('ouro', 'huginn')}
    rng = random.Random(202610291)
    rng.shuffle(conversations)
    candidates = []
    seen = set()
    for item in conversations:
        turns = item.get('conversations', [])
        positions = [i for i, turn in enumerate(turns) if turn.get('from') in ('human', 'user')]
        if not positions:
            continue
        end = rng.choice(positions)
        messages = []
        for turn in turns[:end + 1]:
            role = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant', 'system': 'system'}.get(turn.get('from'))
            if role and isinstance(turn.get('value'), str):
                messages.append(dict(role=role, content=turn['value']))
        if not messages or messages[-1]['role'] != 'user':
            continue
        checksum = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if checksum in seen:
            continue
        ids = {}
        try:
            for model, tokenizer in tokenizers.items():
                tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
                ids[model] = list(tokens['input_ids'] if isinstance(tokens, dict) else tokens)
        except Exception:
            continue
        if min(map(len, ids.values())) < 32 or max(map(len, ids.values())) > 3072:
            continue
        seen.add(checksum)
        candidates.append(dict(prompt_id=str(item.get('id', checksum)), prompt_sha256=checksum,
                               messages=messages, input_ids=ids,
                               input_lengths={model: len(tokens) for model, tokens in ids.items()}))
        if len(candidates) >= 2400:
            break
    if len(candidates) < 700:
        raise RuntimeError('Insufficient distinct supported real-text conversations')
    calibration = candidates[:24]
    correctness = candidates[24:64]
    w1 = candidates[64:264]
    remaining = candidates[264:]
    long = [row for row in remaining if min(row['input_lengths'].values()) >= 1024]
    short = [row for row in remaining if max(row['input_lengths'].values()) <= 768]
    if len(long) < 100 or len(short) < 200:
        raise RuntimeError(f'Insufficient natural text bins: long={len(long)}, short={len(short)}')
    sets = dict(calibration=calibration, correctness=correctness, W1=w1,
                W2=long[:100], W3=short[:80] + long[100:120], W4=short[80:160])
    if len(sets['W3']) != 100:
        raise RuntimeError('W3 must contain 20 percent long prompts')
    for name, values in sets.items():
        if name == 'W3':
            random.Random(202610294).shuffle(values)
        write(ROOT / 'data' / (name + '.json'), values)
    provenance = dict(dataset=repo, revision=info.sha, filename=filename,
                      source_SHA256=hashlib.sha256(open(source, 'rb').read()).hexdigest(),
                      cache_path=source, license='apache-2.0 as declared by the dataset card',
                      dataset_card='https://huggingface.co/datasets/' + repo,
                      selection_seed=202610291, selected_pool=len(candidates),
                      counts={name: len(values) for name, values in sets.items()},
                      rule='One uniformly selected user turn per shuffled conversation. Preserve full preceding conversation. Filter 32..3072 model-chat tokens. No filler or B-dependent selection.',
                      limitation='Filtered English ShareGPT conversation subset. It is not a production arrival trace or an unfiltered population sample.')
    write(ROOT / 'configs/dataset_provenance.json', provenance)
    print('PREPARED', provenance['counts'], flush=True)


if __name__ == '__main__':
    main()
