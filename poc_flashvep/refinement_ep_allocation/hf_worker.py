"""Unmodified official HF generation. One visible physical GPU per worker."""
import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .common import MODEL, REVISION, ROOT, append, rows, digest, reference_helpers,write,snapshot_driver

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--gpu', type=int, required=True, choices=range(8))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(a.gpu)
    assert 0 <= a.shard < a.shards
    ref = reference_helpers()
    chosen = rows(a.manifest)
    if a.limit is not None:
        chosen = chosen[:a.limit]
    chosen = [r for i, r in enumerate(chosen) if i % a.shards == a.shard]
    output = ROOT/'hf'/a.run_id/f'shard{a.shard}.jsonl'
    completed = {r['sample_id'] for r in rows(output)} if output.exists() else set()
    if all(r['sample_id'] in completed for r in chosen):
        print('ALL_SAMPLES_COMPLETE', flush=True)
        return
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
        dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    import inspect
    active_source=inspect.getfile(type(model))
    write(output.parent/f'worker{a.shard}.metadata.json',dict(
        driver=snapshot_driver(__file__),model_source=active_source,model_source_sha256=digest(active_source),
        model_revision=REVISION,gpu=a.gpu,manifest=a.manifest,manifest_sha256=digest(a.manifest),
        torch=str(torch.__version__),cuda=torch.version.cuda,attention=model.config._attn_implementation,
        precision=str(next(model.parameters()).dtype),time=time.time()))
    original = model.forward
    nfe = [0]
    def counted(*args, **kwargs):
        nfe[0] += 1
        return original(*args, **kwargs)
    model.forward = counted
    append(ROOT/'commands.jsonl', dict(kind='HF_START', run_id=a.run_id, gpu=a.gpu,
        manifest=a.manifest, manifest_sha256=digest(a.manifest), shard=a.shard,
        pid=os.getpid(), time=time.time(), revision=REVISION))
    for row in chosen:
        if row['sample_id'] in completed:
            continue
        prompt = tokenizer.apply_chat_template([{'role':'user','content':row['raw_prompt']}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template([{'role':'user','content':row['raw_prompt']}],
            tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda:0')
        seed = 20260915 + row['id']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        nfe[0] = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(inputs=ids, temperature=0., threshold=.95,
                block_length=32, steps=32, gen_length=row['gen_length_cap'], eos_early_stop=True)
        torch.cuda.synchronize()
        wall = 1000*(time.perf_counter()-start)
        output_ids, had_prompt = ref.normalize_generated_ids(generated, ids)
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        parsed, rule = ref.parse_gsm8k(text)
        gold = row['answer'].split('####')[-1].strip().replace(',','') if row.get('answer') else None
        masks = output_ids.count(tokenizer.mask_token_id)
        eos = tokenizer.eos_token_id in output_ids
        record = dict(row, method='REFERENCE_HF', run_id=a.run_id, seed=seed, revision=REVISION,
            worker_physical_gpu=a.gpu, formatted_prompt=prompt, input_ids=ids[0].tolist(),
            prompt_tokens=ids.shape[1], output_ids=output_ids, generation_tokens=len(output_ids),
            raw_generation=tokenizer.decode(output_ids, skip_special_tokens=False),
            postprocessed_generation=text, original_prompt=row['question'], parsed_answer=parsed,
            parser_rule=rule, ground_truth=gold, correct=(parsed==gold) if gold is not None else None,
            nfe=nfe[0], termination_reason='REMAINING_MASK' if masks else 'EOS' if eos else 'GEN_LENGTH_CAP',
            nfe_definition='All official HF model.forward calls. Full visible prefix recomputation.',
            num_generated_blocks=(ids.shape[1]+max(1,len(output_ids))-1)//32-ids.shape[1]//32+1,
            remaining_mask_tokens=masks, official_return_included_prompt=had_prompt,
            wall_ms=wall, timing_kind='MEASURED_WALL_QUALITY_OBSERVER_NOT_PRIMARY_LATENCY',
            protocol='mini_reference_legacy_matched_v1', torch=str(torch.__version__),
            block_length=32, steps_per_block_max=32, threshold=.95, temperature=0,
            time=time.time())
        append(output, record)
        print(row['sample_id'], 'NFE', nfe[0], 'MS', round(wall,1), 'CORRECT', record['correct'], flush=True)
        del generated, ids
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
