"""Check completed evidence against frozen inputs and execution boundaries."""
from collections import Counter
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    questions=read(ROOT/'results/frozen_quality_questions.json')
    inputs={(r['dataset'],r['id']):r for r in questions}
    assert len(inputs)==130
    quality_root=ROOT/'results/quality_native_v3'
    counts=Counter();keys=set();generated_tokens=0;cap_pairs={}
    for path in sorted(quality_root.glob('*_cap*_wave*.json')):
        lane=path.name.split('_')[1]
        for row in read(path):
            frozen=inputs[(row['dataset'],row['problem_id'])]
            key=(row['dataset'],lane,row['seed'],row['cap'],row['problem_id'])
            assert key not in keys,key
            keys.add(key)
            assert row['prompt_sha256']==frozen['prompt_sha256']
            assert row['prompt_tokens']==len(frozen['prompt_ids'])
            assert row['generated_tokens']==len(row['generated_ids'])<=row['cap']
            assert row['total_positions']==row['prompt_tokens']+row['generated_tokens']<=32768
            assert row['forced'] is False and row['manually_reviewed'] is False
            assert row['top_k']==-1
            assert row['temperature']==(1. if lane=='sampling' else 0.)
            assert row['top_p']==(.7 if lane=='sampling' else 1.)
            assert row['seed'] in (SEEDS if lane=='sampling' else [SEEDS[0]])
            counts[(row['dataset'],lane,row['cap'])]+=1
            generated_tokens+=row['generated_tokens']
            cap_pairs.setdefault((row['dataset'],lane,row['seed'],row['problem_id']),{})[row['cap']]=row['generated_ids']
    for (dataset,lane,cap),count in counts.items():
        expected=(90 if lane=='sampling' else 30) if dataset=='aime24' else 100
        assert count==expected,(dataset,lane,cap,count)
    assert read(quality_root/'guard_status.json')['success']
    assert read(quality_root/'seed_reproducibility.json')['exact_token_reproducibility']
    for row in read(quality_root/'ownership.json'):
        assert row['physical_layers']==48 and row['cache_slots']==192
        assert len({c['data_ptr'] for c in row['cache']})==192
    assert len(read(ROOT/'results/native_quality_r4.json')['final_lanes'])==3
    for name in ['reference_components_v1','stage_transfers_v1','physics/TP4_R4_chunk0_v1']:
        assert read(ROOT/'results'/name/'status.json')['success']
        assert read(ROOT/'results'/name/'guard_status.json')['success']
    for row in read(ROOT/'results/large_evidence_manifest.json')['files']:
        assert sha(row['external_path'])==row['sha256']
        assert sha(ROOT/row['original_relative'])==row['sha256']
    assert read(ROOT/'results/final_burner_restore.json')['verification']=='RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs'
    assert read(ROOT/'results/audit.json')['all_pass']
    assert read(ROOT/'results/public_grader_check.json')['all_pass']
    for row in read(ROOT/'results/checkpoint_files.json')['files']:
        assert sha(MODEL/row['file'])==row['sha256']
    prefix_checks=[]
    for key,caps in cap_pairs.items():
        if set(caps)!={4096,8192}:continue
        a,b=caps[4096],caps[8192]
        common=min(len(a),len(b))
        first=next((i for i in range(common) if a[i]!=b[i]),None)
        prefix_checks.append(dict(dataset=key[0],lane=key[1],seed=key[2],problem_id=key[3],
            compared_tokens=common,same_prefix=first is None,first_divergent_token=first))
    write(ROOT/'results/cap_prefix_reproducibility.json',dict(comparisons=prefix_checks,
        same_prefix_count=sum(r['same_prefix'] for r in prefix_checks),total=len(prefix_checks),
        scope='Diagnostic comparison across two output caps. Batch/prompt/seed remain fixed. Not a candidate parity result.'))
    write(ROOT/'results/final_evidence_check.json',dict(success=True,quality_generations=len(keys),
        quality_generated_tokens=generated_tokens,counts=[dict(dataset=k[0],lane=k[1],cap=k[2],count=v) for k,v in counts.items()],
        scientific_verdict=read(ROOT/'results/verdict.json')['verdict'],
        scope='Input identity, full-lane counts, output shape, masks, guard cleanup, source preservation, and archived hashes. Not a candidate correctness PASS.'))
    print('FINAL_EVIDENCE_CHECK_PASS',len(keys),generated_tokens)


if __name__=='__main__':main()
