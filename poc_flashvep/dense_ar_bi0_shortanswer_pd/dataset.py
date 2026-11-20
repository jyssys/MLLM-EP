"""Replace the declared GovReport stratum without outcome-based resampling."""
import collections
import copy
import hashlib
import json
import random
import struct
import time
import zipfile
from pathlib import Path
from .common import ROOT, PREVIOUS, MODEL, MODEL_REVISION, initialize, write, digest

REVISION = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
SEED = 2026112004
TASKS = ('hotpotqa', 'musique', 'narrativeqa', 'qmsum', 'qasper')


def token_hash(ids):
    return hashlib.sha256(struct.pack('<'+'I'*len(ids), *ids)).hexdigest()


def stats(values):
    import numpy as np
    return dict(n=len(values), mean=float(np.mean(values)), min=min(values), max=max(values),
        **{f'p{p}':float(np.percentile(values, p)) for p in (10,25,50,75,90,95,99)})


def main():
    initialize()
    old_path = PREVIOUS / 'data/longbench_v1_mixed200.json'
    assert digest(old_path) == '1a9e6637bbd4f26e1338533d6735fa981c3ab258e0a59ac39c35b8b22d035553'
    archive = Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots') / REVISION / 'data.zip'
    assert digest(archive) == 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'
    with zipfile.ZipFile(archive) as bundle:
        member = next(n for n in bundle.namelist() if n.endswith('/qasper.jsonl') or n == 'qasper.jsonl')
        raw = [json.loads(line) for line in bundle.read(member).decode().splitlines()]
    indices = random.Random(SEED).sample(range(len(raw)), 40)
    frozen = dict(name='LongBench-ShortAnswer200-v2', revision=REVISION,
        qasper_sampling_seed=SEED, qasper_source_indices=indices,
        qasper_source_ids=[raw[i]['_id'] for i in indices],
        order_policy='Replace each GovReport slot in the frozen v1 order. Preserve the relative order and every token of the other160 requests.',
        output_contract=dict(greedy=True, thinking=False, natural_EOS=True, cap=2048),
        input_contract=dict(model_context=40960, output_reserve=2048, input_limit=38912,
                            truncation='Head/tail after the Qwen chat template, matching v1.'),
        fallback_task_candidates=['multifieldqa_en', '2wikimqa'],
        fallback_rule='Investigate only if Qasper mean natural output is not below100 tokens. Never replace individual prompts after observing output.',
        numerical_mode='BI0')
    selection = ROOT / 'data/shortanswer200_v2_sampling_frozen.json'
    if selection.exists():
        assert json.loads(selection.read_text()) == frozen
    else:
        write(selection, frozen)
    templates = json.loads((PREVIOUS/'references/longbench_v1_dataset2prompt.json').read_text())
    write(ROOT/'references/longbench_v1_dataset2prompt.json', templates)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    new = []
    for index in indices:
        source = raw[index]
        prompt = templates['qasper'].format(**source)
        ids = tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        original = len(ids)
        original_hash = token_hash(ids)
        if original > 38912:
            ids = ids[:19456] + ids[-19456:]
        new.append(dict(prompt_id=f'longbench_v1_qasper_{source["_id"]}',task='qasper',category='QA',
            source_dataset='zai-org/LongBench',source_revision=REVISION,source_id=source['_id'],source_index=index,
            original_context=source['context'],original_input=source['input'],reference_answers=source['answers'],
            rendered_prompt_full=prompt,prompt_text_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),
            context_SHA256=hashlib.sha256(source['context'].encode()).hexdigest(),
            input_tokens_original=original,input_tokens=len(ids),input_ids=dict(qwen8=ids),
            input_ids_SHA256=token_hash(ids),original_token_ids_SHA256=original_hash,
            generated_output_tokens=None,output_cap=2048,ignore_eos=False,thinking_disabled=True))
    before = json.loads(old_path.read_text())
    rows = []
    iterator = iter(new)
    for old in before:
        row = next(iterator) if old['task'] == 'gov_report' else copy.deepcopy(old)
        row['sample_id'] = old['sample_id']
        rows.append(row)
    assert len(rows) == len({r['prompt_id'] for r in rows}) == 200
    assert collections.Counter(r['task'] for r in rows) == {t:40 for t in TASKS}
    retained = [r for r in rows if r['task'] != 'qasper']
    assert retained == [r for r in before if r['task'] != 'gov_report']
    path = ROOT / 'data/longbench_shortanswer200_v2.json'
    if path.exists():
        assert json.loads(path.read_text()) == rows
    else:
        write(path, rows)
    # Task/length coverage is selected before reading model outputs.
    small = [next(r for r in rows if r['task']==task) for task in TASKS]
    for row in sorted(rows, key=lambda r:r['input_tokens']) + sorted(rows,key=lambda r:-r['input_tokens']):
        if row not in small:
            small.append(row)
        if len(small) == 8:
            break
    for count, name, selected in ((8,'qualification8.json',small), (32,'qualification32.json',rows[:32])):
        assert len(selected) == count
        write(ROOT/'data'/name, selected)
    truncation = [dict(prompt_id=r['prompt_id'],task=r['task'],original_tokens=r['input_tokens_original'],
                       retained_tokens=r['input_tokens'],policy='head19456/tail19456 after chat template')
                  for r in rows if r['input_tokens_original'] > r['input_tokens']]
    write(ROOT/'data/shortanswer200_v2_truncation_manifest.json',dict(requests=truncation,exclusions=0,
        retained_v1_requests=160,scope='No new length filter or filler. Sampling precedes output and performance.'))
    summary = dict(time=time.time(),dataset_name=frozen['name'],dataset_SHA256=digest(path),
        parent_dataset_SHA256=digest(old_path),sampling_manifest_SHA256=digest(selection),
        source_revision=REVISION,model_revision=MODEL_REVISION,qasper_seed=SEED,
        task_counts=dict(collections.Counter(r['task'] for r in rows)),
        input=stats([r['input_tokens'] for r in rows]),
        by_task={t:stats([r['input_tokens'] for r in rows if r['task']==t]) for t in TASKS},
        truncated_requests=len(truncation),excluded_requests=0,
        preserved_v1_input_ids_160=True,output_distribution='NOT_GENERATED',
        evidence='PRE_SERVING_DATASET_CONTRACT',
        scope='Long-context short-answer research distribution, not general production traffic. QMSum can generate more than100 tokens.')
    write(ROOT/'analysis/shortanswer200_v2_pre_serving_contract.json', summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
