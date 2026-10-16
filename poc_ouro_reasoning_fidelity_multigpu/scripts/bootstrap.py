"""Freeze the evaluation protocol before new generation or scoring."""
import datetime
import subprocess
import urllib.request
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    assert not (ROOT / 'protocol_lock.json').exists()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    original = read(PREVIOUS / 'results/frozen_quality_questions.json')
    questions = []
    for row in original:
        value = dict(row)
        value['arms'] = {}
        for arm, thinking in [('P0', False), ('P1', True)]:
            text = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])],
                tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            ids = tokenizer.encode(text, add_special_tokens=False)
            value['arms'][arm] = dict(text=text, ids=ids,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                last32_ids=ids[-32:], last32_decoded=tokenizer.decode(ids[-32:], skip_special_tokens=False))
        assert value['arms']['P1']['text'] == value['arms']['P0']['text'] + '<think>\n'
        assert value['arms']['P1']['ids'][-2:] == [3, 198]
        questions.append(value)
    write(ROOT / 'results/frozen_questions.json', questions)
    model_files = {p.name: sha(p) for p in MODEL.iterdir() if p.is_file() and p.suffix in ['.json', '.py', '.jinja', '.md']}
    archive = ROOT / 'sources'
    archive.mkdir(parents=True, exist_ok=True)
    urls = {
        'paper.html': 'https://arxiv.org/html/2510.25741v2',
        'project.html': 'https://ouro-llm.github.io/',
        'discussion7.html': 'https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/discussions/7',
        'pinned_card.md': f'https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/raw/{REVISION}/README.md',
        'original_repository.html': 'https://github.com/ByteDance/Ouro'}
    ledger = []
    for name, url in urls.items():
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                data, status = response.read(), response.status
            (archive / name).write_bytes(data)
            ledger.append(dict(url=url, status=status, path=name, sha256=sha(archive / name)))
        except Exception as error:
            ledger.append(dict(url=url, error=repr(error)))
    write(archive / 'manifest.json', ledger)
    write(ROOT / 'results/environment.json', dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        assigned_gpu_query=subprocess.check_output(['nvidia-smi', '--id=4,5,6,7', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv'], text=True),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv'], text=True),
        model_files=model_files, spec_sha256=sha(SPEC)))
    (archive / 'authoritative_spec.md').write_bytes(SPEC.read_bytes())
    lock = dict(version=1, frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=str(MODEL), revision=REVISION, model_files_sha256=model_files,
        physical_gpus=GPUS, logical_mapping=dict(zip(range(4), GPUS)), precision='BF16',
        recurrent_depth=4, early_exit_threshold=1.0, context_limit=32768,
        tokenizer_reference='Transformers 4.57.1 fast tokenizer. Frozen IDs are input to all engines.',
        inputs_sha256=sha(ROOT / 'results/frozen_questions.json'),
        primary_prompt='P1: original frozen instruction + official enable_thinking=True. Selection is source-based, not score-based.',
        diagnostic_prompt='P0: same user content without explicit thinking prefix. Not a competing score-selected primary.',
        diagnostic_ids=[r['id'] for r in questions[:8]],
        fidelity_calibration_ids=[r['id'] for r in questions[:4]],
        fidelity_confirmation_ids=[r['id'] for r in questions[4:8]],
        fidelity_teacher_checkpoints=[0, 1, 4, 32, 128, 1024],
        fidelity_natural_cap=256, fidelity_long_cap=2048,
        fidelity_tolerance_rule='Use calibration only. Freeze confirmation limits before reading confirmation differences. Cross-backend BF16 need not be bitwise equal.',
        cross_cap_regression=dict(dataset='math100', id='test/algebra/1339.json',
            caps=[256,512], repeats=2, batch='original previous-run B8 wave', purpose='diagnostic, not accuracy'),
        quality=dict(datasets={'aime24':'all 30, three sampled seeds plus separate greedy', 'math100':'same frozen 100-question subset, greedy'},
            temperature=1.0, top_p=.7, top_k=-1, repetition_penalty=1., seeds=SEEDS,
            greedy_temperature=0., eos_enabled=True, eos_token_id=2, stop_strings=[],
            caps=[8192,16384,24576], extension_trigger='At least 10% length stops in a complete lane. Repeat the entire lane.',
            maximum_total_positions=32768, fixed_batch_size=4, graph_sizes=[1,2,4],
            quality_ready_completion_target=.90, capped_outputs_count_as_completed_correct=False,
            parser='Reuse balanced final boxed answer parser and strict Math-Verify 0.8.0. No manual score changes.',
            bootstrap_unit='problem, retain all sampled seeds', bootstrap_draws=5000),
        systems_gate='No broad real-workload campaign before credible quality and resolved consequential runtime errors.',
        exploratory_if_blocked='Small labeled native real-workload audit only. No new framework or speedup claim.',
        optional_10_seeds=False, optional_full_math500=False, method_implementation=False)
    write(ROOT / 'protocol_lock.json', lock)
    print('Frozen protocol:', sha(ROOT / 'protocol_lock.json'), flush=True)

if __name__ == '__main__':
    main()
