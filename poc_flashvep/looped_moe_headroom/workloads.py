"""Freeze raw text before model comparisons. All models receive the same text."""
import hashlib
import json
from pathlib import Path
import random
from .common import ROOT, SEED, digest, write

SOURCE = Path('/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/dataset_manifest.json')
PILE = Path('/home/esjung/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots/127bfedcd5047750df5ccf3a12979a47bfa0bafa/data/train-00000-of-00001-4746b8785c874cc7.parquet')

def main():
    import pyarrow.parquet as pq
    target = ROOT/'workload_manifest.json'
    assert not target.exists(), 'Frozen workload cannot be overwritten'
    source = json.loads(SOURCE.read_text())
    generator = random.Random(SEED)
    rows = []
    groups = {
        'factual': ['Wikipedia (en)', 'PubMed Abstracts', 'ArXiv'],
        'narrative': ['Gutenberg (PG-19)', 'Books3', 'BookCorpus2', 'OpenSubtitles', 'YoutubeSubtitles', 'Enron Emails', 'HackerNews'],
        'procedural': ['StackExchange', 'OpenWebText2', 'Pile-CC'],
        'code': ['Github'],
    }
    pile = pq.read_table(PILE).to_pylist()
    for group, domains in groups.items():
        candidates = [(i,r) for i,r in enumerate(pile) if r['meta']['pile_set_name'] in domains and len(r['text'])>=1024]
        generator.shuffle(candidates)
        assert len(candidates)>=42, (group,len(candidates))
        for j,(index,row) in enumerate(candidates[:42]):
            rows.append(dict(id=f'pile/{index}', dataset='language', category=group,
                source_domain=row['meta']['pile_set_name'], split='smoke' if j<2 else 'calibration' if j<10 else 'heldout',
                text=row['text'][:20000], gold=None))
    for domain in ['gsm8k','math500']:
        selected = [r for r in source['rows'] if r['dataset']==domain]
        generator.shuffle(selected)
        for j,row in enumerate(selected[:168]):
            rows.append(dict(id=row['id'],dataset=domain,category=domain,
                split='smoke' if j<8 else 'calibration' if j<40 else 'heldout',
                text=('Question: ' if domain=='gsm8k' else 'Problem: ')+row['question']+'\nAnswer:',
                question=row['question'],gold=row['answer']))
    for i,text in enumerate(['The same word repeats. '*1000,
            ''.join(f'item_{j} = {j} + {j+1}\n' for j in range(1000)),
            'red blue green yellow. '*1000,
            'Copy these values exactly: 1 7 3 9 2 8. '*1000]):
        rows.append(dict(id=f'stress/{i}',dataset='synthetic',category='mechanism',split='smoke',text=text,gold=None))
    for row in rows:
        row['text_sha256']=hashlib.sha256(row['text'].encode()).hexdigest()
    write(target,dict(seed=SEED, rows=rows, source=str(SOURCE), source_sha256=digest(SOURCE),
        sources=dict(pile_repository='NeelNanda/pile-10k',pile_revision='127bfedcd5047750df5ccf3a12979a47bfa0bafa',
            pile_sha256=digest(PILE), gsm_repository=source['gsm_repository'],gsm_revision=source['gsm_revision'],
            math_repository=source['math_repository'],math_revision=source['math_revision']),
        protocol=dict(language_heldout_per_model=128,language_calibration=32,
            reasoning_available_heldout_per_domain=128,reasoning_first_wave_per_domain=8,
            reasoning_first_wave_cap=256, reasoning_expansion='Expand only if NLL and generation evidence justify a quality-preservation claim.',
            generation_fidelity_cap=16,teacher_forced_lengths=[128,512,2048],
            baseline_generation_lengths=[16,64],generation_prefix_lengths=[128,512],
            greedy=True, stop_eos=True, cache='official full-prefix recomputation',
            downstream_claim='First-wave samples are underpowered. Truncated answers do not establish model accuracy.',
            training_overlap='Dataset selection is independent of tuning. Training overlap is not audited.',
            category_limit='Category labels describe source domains, not certified document genres.')))
    print({d:sum(r['dataset']==d for r in rows) for d in ['language','gsm8k','math500','synthetic']})

if __name__=='__main__':
    main()
