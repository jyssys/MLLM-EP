"""Freeze data and placement rules before new experimental outcomes."""
import datetime
import gzip
import hashlib
import random
import subprocess
from poc_depth_cdb_factorial.contract import *

def main():
    assert not (ROOT/'artifacts/arm_manifest.json').exists()
    for name in ['artifacts', 'reports', 'figures', 'outputs']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    (ROOT/'reports/spec.md').write_bytes(SPEC.read_bytes())
    original = '/home/esjung/MLLM-EP-github'
    def cmd(args, cwd=original):
        return subprocess.check_output(args, cwd=cwd)
    continuity = dict(base_commit=BASE_COMMIT,
        original_staged_sha256=hashlib.sha256(cmd(['git','diff','--cached','--binary'])).hexdigest(),
        original_unstaged_sha256=hashlib.sha256(cmd(['git','diff','--binary'])).hexdigest(),
        spec_sha256=sha(SPEC), cdb_revision=cmd(['git','rev-parse','HEAD'],CDB).decode().strip(),
        sources={str(p.relative_to(PROJECT)):sha(p) for p in [
            V3_ROOT/'reports/final_report.md', V3_ROOT/'reports/decode_oracle.md',
            V3_ROOT/'reports/live_dp4_baselines.md', V3_ROOT/'scripts/live_worker.py',
            V3_ROOT/'scripts/live_dp.py', OLD_ROOT/'runtime.py']})
    write(ROOT/'artifacts/continuity.json', continuity)
    old_plan = read(V3_ROOT/'artifacts/live_clean_plan.json')
    ids = next(c['ids'] for c in old_plan['cases'] if c['name']=='clean_burst_C64_round_robin')
    discovery = read(V3_ROOT/'artifacts/discovery_frozen.json')
    byid = {r['id']:r for r in discovery['rows']}
    speed = [byid[rid] for rid in ids]
    quality_source = read(OLD_ROOT/'artifacts/quality_protocol_v3_pilot32.json')
    quality = []
    for domain in ['gsm8k','math500']:
        candidates = [r for r in quality_source['rows'] if r['dataset']==domain]
        random.Random(20261019).shuffle(candidates)
        quality += candidates[:128]
    assert len(quality)==256 and len({r['id'] for r in quality})==256
    write(ROOT/'artifacts/workload_frozen.json', dict(speed=speed, quality=quality,
        speed_cap=256, quality_cap=8192, quality_source_sha256=sha(OLD_ROOT/'artifacts/quality_protocol_v3_pilot32.json'),
        discovery_source_sha256=sha(V3_ROOT/'artifacts/discovery_frozen.json'),
        quality_selection='Seed 20261019 shuffles each complete domain independently. First 128 per domain.',
        quality_scope='9.70% of GSM8K test and 25.60% of MATH-500. Not full-benchmark evidence.',
        speed_scope='Historical held-out request IDs, reused. Not fresh held-out.',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    depth_work = {}
    with gzip.open(V3_ROOT/'artifacts/decode_depth_traces.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line)
            rid=r.get('id',r.get('request_id'))
            if rid is not None:
                depth_work[rid]=depth_work.get(rid,0)+r.get('depth',r.get('exit_depth_executed',0))
    # The request aggregate is the authoritative sum if the event schema differs.
    summaries = read(V3_ROOT/'artifacts/request_depth_summary.json')
    if isinstance(summaries,dict): summaries=summaries.get('rows',summaries.get('requests',[]))
    for r in summaries:
        rid=r.get('id',r.get('request_id'))
        if rid in byid:
            depth_work[rid]=r.get('total_loop_work',r.get('actual_loop_work',r.get('loop_work',depth_work.get(rid,0))))
    assert all(depth_work.get(rid,0)>0 for rid in ids), list(depth_work.items())[:2]
    write(ROOT/'artifacts/speed_reference_depth_work.json',depth_work)
    manifest=dict(arms=ARMS, model=MODEL_ID,model_revision=REVISION,cdb_revision=CDB_REVISION,
        BF16=True,q=.5,min_R=2,max_R=4,kv='last_exited',prefill='full R4',temperature=0,
        EOS=0, graphs_decode=True,graphs_prefill=False,async_batching=False,delayed_gate=False,
        local_batch_cap=16,kv_blocks=2048,kv_block_rows=16,max_model_len=10240,
        per_rank_KV_GiB=24.0,primary_repeats=3,concurrency=[4,8,16,32,64],
        generic='Queue-count placement at simultaneous burst admission. This reduces to rank round-robin with stable ties.',
        hindsight='Deterministic LPT of recorded total decode recurrent work. Stable ties by incoming request index and rank.',
        hindsight_quality_reference='Earlier A0 measurement, frozen before A2/A4. Never score answers for placement.',
        hindsight_sensitivity='Reuse v3 trace-derived selected C64 placement, unchanged, in A2/A4.',
        quality_ids=[r['id'] for r in quality],speed_ids=ids,seeds=[20261019,20261020,20261021],
        paired_repeats='Identical cohort and deterministic order. Alternate/refill arm order across repeats.',
        goodput='Request completed within max(10 s, 200 ms per emitted token). Completion here includes capped bounded progress.',
        CI='2000 paired cohort-repeat resamples for time ratios. Paired question resamples for accuracy.',
        frozen_before_new_inference=True)
    write(ROOT/'artifacts/arm_manifest.json',manifest)
    cases=[]
    for repeat in range(3):
        for C in [4,8,16,32,64]:
            group=ids[:C]
            for arm in (['A0','A1','A2','A3','A4'] if repeat%2==0 else ['A4','A3','A2','A1','A0']):
                placement=lpt(group,depth_work) if ARMS[arm]['hindsight'] else {rid:i%4 for i,rid in enumerate(group)}
                if arm=='A0':placement={rid:0 for rid in group}
                cases.append(dict(name=f'speed_C{C}_{arm}_r{repeat}',kind='speed',arm=arm,C=C,repeat=repeat,
                    ids=group,cap=256,placements=placement))
    write(ROOT/'artifacts/speed_plan.json',dict(cases=cases,local_batch_cap=16))
    smoke_ids=ids[:8]
    smoke=[dict(name=f'smoke_{a}',kind='smoke',arm=a,C=8,repeat=0,ids=smoke_ids,cap=16,
        placements=({rid:0 for rid in smoke_ids} if a=='A0' else lpt(smoke_ids,depth_work)
            if ARMS[a]['hindsight'] else {rid:i%4 for i,rid in enumerate(smoke_ids)})) for a in ARMS]
    write(ROOT/'artifacts/smoke_plan.json',dict(cases=smoke,local_batch_cap=16))
    # Four disjoint 64-request quality cohorts, with each domain interleaved.
    quality_order=[r['id'] for pair in zip(quality[:128],quality[128:]) for r in pair]
    qcases=[dict(name=f'quality_A0_part{k}',kind='quality',arm='A0',C=64,repeat=0,
        ids=quality_order[k*64:(k+1)*64],cap=8192,
        placements={rid:0 for rid in quality_order[k*64:(k+1)*64]}) for k in range(4)]
    write(ROOT/'artifacts/quality_A0_plan.json',dict(cases=qcases,local_batch_cap=16))
    print('FROZEN',len(cases),'speed cases and',len(quality),'quality questions.')

if __name__=='__main__':main()
