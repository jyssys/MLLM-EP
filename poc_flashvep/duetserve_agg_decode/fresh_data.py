"""Freeze a new real-text confirmation pool before using its performance."""
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

from .common import MODEL, MODEL_REVISION, PRIOR, ROOT, REPO, digest, initialize, write


def main():
    initialize()
    target = ROOT / 'data/fresh_manifest.json'
    if target.exists():
        print('FRESH_POOL_ALREADY_FROZEN')
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    seen, prefixes = set(), set()
    prior_manifest = json.loads((PRIOR / 'data/unseen_manifest.json').read_text())
    paths = {REPO / row['path'] for row in prior_manifest['prior_files']}
    paths.update(PRIOR / 'data' / name for name in ('discovery.json', 'qualification.json',
        'calibration.json', 'unseen_confirmation.json', 'unseen_calibration.json'))
    provenance = []
    for path in sorted(paths):
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            continue
        for row in value:
            if not isinstance(row, dict) or 'prompt_id' not in row:
                continue
            seen.add(row['prompt_id'])
            if 'qwen8' in row.get('input_ids', {}):
                prefixes.add(tuple(row['input_ids']['qwen8'][:7900]))
        provenance.append(dict(path=str(path), SHA256=digest(path)))
    archive = Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip')
    candidates = []
    with zipfile.ZipFile(archive) as stream:
        for task in ('gov_report', 'qmsum', 'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique'):
            names = [name for name in stream.namelist() if name.endswith('/' + task + '.jsonl') or name == task + '.jsonl']
            if not names:
                continue
            for index, line in enumerate(stream.read(names[0]).decode().splitlines()):
                row = json.loads(line)
                identifier = f'longbench_{task}_{row.get("_id", index)}'
                if identifier in seen:
                    continue
                messages = [dict(role='user', content='Document:\n' + row['context'] + '\nQuestion:\n' + row.get('input', ''))]
                candidates.append((identifier, task, messages))
    random.Random(2026111621).shuffle(candidates)
    tail = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
    check = tokenizer.apply_chat_template([dict(role='user', content='test')], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert check[-len(tail):] == tail
    rows = []
    for identifier, task, messages in candidates:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        if len(ids) < 8000:
            continue
        clipped = ids[:8000 - len(tail)] + tail
        if tuple(clipped[:7900]) in prefixes:
            continue
        prefixes.add(tuple(clipped[:7900]))
        rows.append(dict(prompt_id=identifier, source_dataset='LongBench/' + task,
            source_SHA256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            input_ids={'qwen8': clipped}, length=8000, original_tokenized_length=len(ids),
            source_type='CONTROLLED_PUBLIC_DOCUMENT_PREFIX_WITH_COMPLETE_GENERATION_SUFFIX',
            truncation_explicit=True, thinking_disabled_suffix_verified=True))
        if len(rows) == 128:
            break
    assert len(rows) == 128
    write(ROOT / 'data/fresh_confirmation.json', rows)
    write(target, dict(time=time.time(), model_revision=MODEL_REVISION, selection_seed=2026111621,
        dataset_revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672', archive_SHA256=digest(archive),
        prior_files=provenance, excluded_prompt_ID_count=len(seen), prior_prompt_ID_overlap=0,
        prior_7900_token_prefix_overlap=0, generation_suffix_ids=tail,
        pool_SHA256=digest(ROOT / 'data/fresh_confirmation.json'), prompt_ids=[r['prompt_id'] for r in rows],
        scope='Exclude the predecessor unseen pool and its prior selection inputs. Declared duration cycles reuse these already excluded source pools. Confirmation is not used for parameter tuning.'))
    print('FRESH_POOL_FROZEN', len(rows))


if __name__ == '__main__':
    main()
