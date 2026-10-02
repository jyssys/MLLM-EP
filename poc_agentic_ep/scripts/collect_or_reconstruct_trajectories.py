"""Freeze genuine public SWE-agent histories before observing any routes."""
import gzip
import hashlib
import json
import random
import requests
from transformers import AutoTokenizer
from poc_agentic_ep.contract import ROOT, SPEC, TARGET, TARGET_REV, write, sha, lcp, observation_frontier

DATASET = 'SWE-bench/SWE-smith-trajectories'
REVISION = '08e109b4a59eaeebf80e4675cd125d42e7ac99a4'
SEED = 3407


def main():
    root = ROOT/'results'; root.mkdir(parents=True, exist_ok=True)
    if (root/'frozen_manifest.json').exists(): raise FileExistsError('frozen cohort already exists')
    session = requests.Session()
    meta = session.get('https://huggingface.co/api/datasets/'+DATASET, timeout=40)
    meta.raise_for_status(); assert meta.json()['sha'] == REVISION
    response = session.get('https://datasets-server.huggingface.co/rows', params=dict(
        dataset=DATASET, config='default', split='xml', offset=0, length=100), timeout=90)
    response.raise_for_status()
    assert response.headers['x-revision'] == REVISION, 'viewer revision changed'
    candidates = response.json()['rows']
    with gzip.open(root/'source_candidates.json.gz','wt') as f: json.dump(candidates,f)
    tokenizer = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    order = list(range(len(candidates))); random.Random(SEED).shuffle(order)
    selected = []; rejected = []; used_instances = set(); rows = []; raw = []
    for idx in order:
        value = candidates[idx]
        if 'messages' in value.get('truncated_cells',[]):
            rejected.append(dict(row=idx, reason='VIEWER_TRUNCATED_MESSAGES')); continue
        sample = value['row']
        if sample['instance_id'] in used_instances: continue
        messages = json.loads(sample['messages'])
        assistant_indices = [i for i,m in enumerate(messages) if m['role']=='assistant']
        if len(assistant_indices)<9:
            rejected.append(dict(row=idx,reason='FEWER_THAN_NINE_TURNS')); continue
        turns = []; prev_prompt = []; prev_completed = []
        for turn, end in enumerate(assistant_indices[:9]):
            history = messages[:end]
            if any(m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str) for m in history): break
            prompt = tokenizer.apply_chat_template(history,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids = tokenizer.encode(prompt,add_special_tokens=False)
            continuation = tokenizer.encode(messages[end]['content'],add_special_tokens=False)[:32]
            completed = tokenizer.apply_chat_template(messages[:end+1],tokenize=True,add_generation_prompt=False,enable_thinking=False)
            if hasattr(completed,'keys'): completed = completed['input_ids']
            ordinary_lcp = lcp(prev_prompt,ids)
            completed_lcp = lcp(prev_completed,ids)
            # Stronger exclusion: previous recorded assistant completion is not
            # counted again as a newly arriving tool observation.
            observation_start=observation_frontier(history,tokenizer,ids) if turn else 0
            frontier = max(ordinary_lcp,completed_lcp,observation_start) if turn else 0
            turns.append(dict(trajectory_id=sample['traj_id'],instance_id=sample['instance_id'],
                turn=turn,request_id=f'traj{len(selected):02d}_turn{turn:02d}',
                prompt_token_ids=ids,teacher_forced_ids=continuation,input_token_ids=ids+continuation,
                prompt_tokens=len(ids),decode_tokens=len(continuation),
                common_prompt_prefix=ordinary_lcp,previous_completed_prefix=completed_lcp,
                current_observation_frontier=observation_start,
                exclusion_start=frontier,new_suffix_tokens=len(ids)-frontier,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
            prev_prompt=ids;prev_completed=list(completed)
        if len(turns)!=9 or max(len(t['input_token_ids']) for t in turns)>16384 or min(t['new_suffix_tokens'] for t in turns)<8:
            rejected.append(dict(row=idx,reason='BOUNDED_CONTEXT_OR_NONEMPTY_SUFFIX_REQUIREMENT'));continue
        split='calibration' if len(selected)<4 else 'heldout'
        for t in turns:t['split']=split
        selected.append(dict(trajectory_id=sample['traj_id'],instance_id=sample['instance_id'],
            source_row=idx,split=split,turns=9,max_context=max(len(t['input_token_ids']) for t in turns)))
        rows.extend(turns);used_instances.add(sample['instance_id'])
        raw.append(dict(trajectory_id=sample['traj_id'],instance_id=sample['instance_id'],
            messages=messages[:assistant_indices[8]+1]))
        if len(selected)==12:break
    if len(selected)!=12:raise RuntimeError(f'cohort shortage: {len(selected)}/12; do not lower validity thresholds')
    with gzip.open(root/'frozen_messages.json.gz','wt') as f:json.dump(raw,f)
    write(root/'frozen_manifest.json',dict(dataset=DATASET,dataset_revision=REVISION,source_split='xml',
        source_endpoint=response.url,source_sha256=sha(root/'source_candidates.json.gz'),seed=SEED,
        selection='seeded permutation of first100 public rows; metadata-only eligibility; no outcome/route selection',
        trajectories=selected,rows=rows,rejected=rejected,target_revision=TARGET_REV,
        tokenizer_hashes={n:sha(Path(TARGET)/n) for n in ('tokenizer.json','tokenizer_config.json')},
        protocol='Original semantic system/action/observation messages; native Qwen3 template, thinking=False; max context16384; teacher-forced recorded continuation32.',
        exclusion='max(LCP(previous prompt,current prompt),LCP(previous recorded completed conversation,current prompt)); shared history never in primary metrics',
        decode_lane='Teacher-forced recorded assistant32, processed in causal prefill; not AR timing/agent quality',
        spec_path=str(SPEC),spec_sha256=sha(SPEC)))
    print(json.dumps(dict(trajectories=len(selected),turns=len(rows),heldout_adjacent_pairs=8*8,
        max_context=max(len(t['input_token_ids']) for t in rows),new_suffix_total=sum(t['new_suffix_tokens'] for t in rows))))


if __name__=='__main__':
    from pathlib import Path
    main()
