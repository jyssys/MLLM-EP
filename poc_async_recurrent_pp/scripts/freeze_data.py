"""Freeze calibration, smoke, pilot, and full evaluation IDs before inference."""
import hashlib
import urllib.request
import sys
sys.path.insert(0, '/home/esjung/external/async-recurrent-pp-assets/cpu_deps')
import pyarrow.parquet as pq
from poc_async_recurrent_pp.contract import *


def ordered(rows):
    return sorted(rows, key=lambda row: hashlib.sha256(('PARPP-v1:'+row['id']).encode()).hexdigest())


def main():
    target = ROOT/'outputs/manifests/data.json'
    assert not target.exists()
    base = Path('/home/esjung/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main')
    def gsm(split):
        return [dict(id=f'gsm8k/{split}/{i}', dataset='gsm8k', question=row['question'],
                     answer=row['answer'].split('####')[-1].strip())
                for i,row in enumerate(pq.read_table(base/f'{split}-00000-of-00001.parquet').to_pylist())]
    revision = '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'
    url = f'https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{revision}/test.jsonl'
    ASSETS.mkdir(exist_ok=True, parents=True)
    local = ASSETS/'math500.jsonl'
    if not local.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            local.write_bytes(response.read())
    math = [dict(id=f'math500/test/{i}', dataset='math500', question=row['problem'], answer=row['answer'])
            for i,row in enumerate(json.loads(line) for line in local.read_text().splitlines())]
    test = gsm('test')
    calibration = ordered(gsm('train'))[:8]
    pilot = ordered(test)[:64] + ordered(math)[:64]
    smoke = [dict(id=f'smoke/{i}', dataset='smoke',question=q,answer=a) for i,(q,a) in enumerate([
        ('What is 2 + 3?', '5'), ('What is 7 times 6?', '42'), ('Solve 2*x + 3 = 11.', '4'),
        ('What is 100 minus 37?', '63'), ('What is the square of 9?', '81'),
        ('Compute 12 divided by 4.', '3'), ('A rectangle is 3 by 5. Find its area.', '15'),
        ('Compute 1/2 + 1/4.', r'\frac{3}{4}'), ('Find the next integer after 199.', '200'),
        ('Solve x - 8 = 5.', '13'), ('What is 15 percent of 200?', '30'),
        ('Find the sum 1 + 2 + 3 + 4.', '10'), ('Find the positive square root of 64.', '8'),
        ('Compute 3 cubed.', '27'), ('Compute 20 - 4 * 3.', '8'), ('What is 0 multiplied by 789?', '0')])]
    write(target, dict(calibration=calibration, smoke=smoke, pilot=pilot, full=test+math,
        gsm_revision='740312add88f781978c0658806c59bc2815b9866', math_revision=revision,
        math_sha256=sha(local), primary_cap=2048, truncation_followup_cap=8192,
        calibration_disjoint=True, split_before_inference=True,
        early_failure_rule='After first 16 pilot items per dataset: stop if every candidate has paired accuracy CI upper bound below -10 pp. Otherwise complete 64 per dataset.',
        prompt_suffix='\nPlease reason step by step and put your final answer in \\boxed{}.',
        warning='Smoke is parser/integrity only. Pilot cannot establish GO. Full 1819 items are required for GO.'))
    print(dict(calibration=len(calibration), smoke=len(smoke),pilot=len(pilot),full=len(test)+len(math),manifest_sha256=sha(target)))


if __name__ == '__main__':
    main()
