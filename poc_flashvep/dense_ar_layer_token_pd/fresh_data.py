"""Select disjoint real documents before new candidate performance is read."""
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path
from .common import MODEL, MODEL_REVISION, PREVIOUS, REPO, ROOT, SLO_PRIOR, digest, write


def main():
    target = ROOT / 'data/heldout_manifest.json'
    if target.exists():
        print('DATA_ALREADY_FROZEN', flush=True)
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    seen, prefixes, provenance = set(), set(), []
    old = json.loads((PREVIOUS / 'data/fresh_manifest.json').read_text())
    paths = {Path(row['path']) for row in old['prior_files']}
    paths.add(PREVIOUS / 'data/fresh_confirmation.json')
    paths.update(SLO_PRIOR / 'data' / n for n in ('discovery.json', 'qualification.json',
        'unseen_calibration.json', 'unseen_confirmation.json'))
    for path in sorted(paths):
        values = json.loads(path.read_text())
        if not isinstance(values, list):
            continue
        for row in values:
            if isinstance(row, dict) and 'prompt_id' in row:
                seen.add(row['prompt_id'])
                ids = row.get('input_ids', {}).get('qwen8')
                if ids:
                    prefixes.add(hashlib.sha256(bytes(str(ids[:7900]), 'utf-8')).hexdigest())
        provenance.append(dict(path=str(path), SHA256=digest(path)))
    archive = Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip')
    candidates = []
    with zipfile.ZipFile(archive) as stream:
        for task in ('gov_report', 'qmsum', 'narrativeqa', 'qasper', 'multifieldqa_en',
                     'hotpotqa', '2wikimqa', 'musique', 'multi_news', 'repobench-p', 'lcc'):
            names = [n for n in stream.namelist() if n.endswith('/' + task + '.jsonl') or n == task + '.jsonl']
            if not names:
                continue
            for index, line in enumerate(stream.read(names[0]).decode().splitlines()):
                raw = json.loads(line)
                identifier = f'longbench_{task}_{raw.get("_id", index)}'
                if identifier in seen:
                    continue
                messages = [dict(role='user', content='Document:\n' + raw['context'] + '\nQuestion:\n' + raw.get('input', ''))]
                candidates.append((identifier, task, messages))
    random.Random(2026111707).shuffle(candidates)
    tail = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
    test = tokenizer.apply_chat_template([dict(role='user', content='test')], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert test[-len(tail):] == tail
    records = []
    for identifier, task, messages in candidates:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        if len(ids) < 8000:
            continue
        clipped = ids[:8000 - len(tail)] + tail
        prefix = hashlib.sha256(bytes(str(clipped[:7900]), 'utf-8')).hexdigest()
        if prefix in prefixes:
            continue
        prefixes.add(prefix)
        records.append(dict(prompt_id=identifier, source_dataset='LongBench/' + task,
            source_SHA256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            original_tokenized_length=len(ids), full_ids=ids[:12288], tail=tail,
            truncation_explicit=True, thinking_disabled_suffix_verified=True))
        if len(records) == 160:
            break
    assert len(records) == 160, 'Insufficient disjoint long real documents'
    def pool(group, variable=False):
        rows = []
        for i, record in enumerate(group):
            length = (2048, 4096, 8192, 12288)[i % 4] if variable else 8000
            if length > len(record['full_ids']):
                length = 8192 if len(record['full_ids']) >= 8192 else 4096
            row = {k: v for k, v in record.items() if k not in ('full_ids', 'tail')}
            row.update(source_prompt_id=record['prompt_id'], prompt_id=record['prompt_id'] + f'_L{length}',
                input_ids=dict(qwen8=record['full_ids'][:length - len(tail)] + tail), length=length)
            rows.append(row)
        return rows
    files = {}
    for name, group, variable in (('new_discovery32_mixed_input.json', records[:32], True),
                                  ('heldout128_8000.json', records[32:], False),
                                  ('heldout128_mixed_input.json', records[32:], True)):
        values = pool(group, variable)
        write(ROOT / 'data' / name, values)
        files[name] = dict(count=len(values), SHA256=digest(ROOT / 'data' / name),
                          prompt_ids=[r['prompt_id'] for r in values])
    write(target, dict(time=time.time(), model_revision=MODEL_REVISION,
        dataset_revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672', archive_SHA256=digest(archive),
        selection_seed=2026111707, prior_files=provenance, excluded_prompt_ID_count=len(seen),
        prior_prompt_ID_overlap=0, prior_7900_prefix_overlap=0,
        discovery_heldout_source_ID_overlap=0, generation_suffix_ids=tail, files=files,
        scope='Controlled real-document prefixes. No filler extension. Fixed and mixed-input heldout pools share the same source128 documents. They are not independent datasets. No candidate performance used for selection.'))
    print('FRESH_DOCUMENTS_FROZEN', len(records), flush=True)


if __name__ == '__main__':
    main()
