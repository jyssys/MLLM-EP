"""Freeze policies, inputs, and historical evidence before any new GPU execution."""
import datetime
import subprocess
from .contract import *

def cmd(args,cwd=PROJECT):return subprocess.check_output(args,cwd=cwd,text=True).strip()

def main():
    assert not (ROOT/'run_manifest.json').exists()
    ROOT.mkdir(parents=True,exist_ok=True)
    for name in ['commands','logs','numerical','raw','timelines','state']: (ROOT/name).mkdir(exist_ok=True)
    (ROOT/'spec_frozen.md').write_bytes(SPEC.read_bytes())
    old=PROJECT/'poc_flashvep/reports/ouro_tp4_parallelism_discovery'
    protected={str(p.relative_to(PROJECT)):sha(p) for folder in ['poc_flashvep/ouro_tp4','poc_flashvep/ouro_request_pp',
        'poc_flashvep/reports/ouro_tp4_parallelism_discovery','poc_flashvep/reports/ouro_request_pp_discovery']
        for p in (PROJECT/folder).glob('*') if p.is_file() and p.suffix in ['.py','.md','.json']}
    write(ROOT/'source_hashes.json',protected)
    source=PROJECT/'poc_decode_first_depth_aware_dp/artifacts/quality_protocol_v3_pilot32.json'
    meta=read(PROJECT/'poc_decode_first_depth_aware_dp/artifacts/frozen_workload.json')
    rows=read(source)['rows'];ordered=[]
    for domain,n in [('gsm8k',64),('math500',50)]:
        group=sorted([r for r in rows if r['dataset']==domain],key=lambda r:int(r['id'].rsplit('/',1)[-1]))
        assert len(group)==(1319 if domain=='gsm8k' else 500)
        ordered.extend([dict(r,pilot=i<n,prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest()) for i,r in enumerate(group)])
    write(ROOT/'dataset_manifest.json',dict(rows=ordered,source=str(source.relative_to(PROJECT)),source_sha256=sha(source),
        gsm_repository='openai/gsm8k',gsm_revision=meta['gsm_revision'],math_repository='HuggingFaceH4/MATH-500',
        math_revision=meta['math_revision'],split='test',selection='First 64/50 numeric canonical question IDs. Metadata only.',
        continuity='Reused public evaluation questions. Not newly held-out research data.',
        max_new_tokens=4096,EOS=True,temperature=0.,stop_strings='Preserved per-request list',
        tokenizer_revision=REVISION,prompt_recipe='CDB three-shot CoT for GSM8K. Zero-shot boxed MATH. Base-model completion, not Thinking/chat.',
        calibration='Synthetic prompts and train-only demonstrations, disjoint from evaluation IDs.',full_counts=dict(gsm8k=1319,math500=500)))
    write(ROOT/'arm_manifest.json',dict(arms=ARMS,model='ByteDance/Ouro-1.4B',revision=REVISION,BF16=True,
        static_R=4,dynamic=dict(reference=CDB,revision=CDB_REV,q=.5,min_R=2,max_R=4,KV='last_exited',prefill='full R4',
            refill=False,policy_choice='Prior validated CDB synchronous no-refill, same policy on every candidate.'),
        natural_C=4,replay_C=[1,4,8,16],primary_repeats=5,accuracy_cap=4096,
        primary_T4='Native vLLM 0.26.0. Engine differences must remain visible.',
        custom_TP='Diagnostic and architecture-controlled exploratory candidate, not replacement of native T4'))
    write(ROOT/'run_manifest.json',dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base_commit=cmd(['git','rev-parse','HEAD']),branch=cmd(['git','branch','--show-current']),
        original_dirty=cmd(['git','status','--porcelain']),spec=str(SPEC),spec_sha256=sha(SPEC),
        CDB_revision=cmd(['git','rev-parse','HEAD'],CDB),model_revision=REVISION,
        push_authority='Final explicit user instruction requests push.',stages={str(i):'NOT_RUN' for i in range(6)}))
    write(ROOT/'environment.json',dict(gpus=cmd(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version','--format=csv']),
        topology=cmd(['nvidia-smi','topo','-m']),NVLink=cmd(['nvidia-smi','nvlink','-s','-i','4,5,6,7']),
        processes=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),
        tokenizer_files={p.name:sha(p) for p in MODEL.iterdir() if p.suffix=='.json'},
        checkpoint_sha256=sha(MODEL/'model.safetensors'),interpreters=[PYTHON,VLLM_PYTHON]))
    (ROOT/'preflight.md').write_text('''# Frozen preflight

Use physical GPUs 4–7 only. Single-GPU jobs use GPU4.
The requested spec path was absent. The same named document exists in the main repository.
`spec_frozen.md` preserves all 426 lines. This task does not modify the main repository.
Previous source/report hashes are frozen in `source_hashes.json`.
The historical C2 failure remains FAIL. It does not cancel this spec's exploratory quality lane.
Before each GPU job, check ownership. Stop only identified project burners.
Restore burners on every permitted GPU that no experiment currently uses.
''')
    (ROOT/'numerical_protocol.md').write_text('''# Prospective numerical protocol

Historical strict C2 retains its original limits and P512 FAIL.
No new tensor-error PASS threshold is defined from test answers.
First compare HF, custom unsharded, custom TP4, native TP1, and native TP4 on synthetic locked prompts.
Use P128/P256/P512/P2048 and inspect all loops/layers at sampled positions.
At P512 retain zero-based steps 18, 19, 26, and 27.
Separate backend differences from same-engine TP1/TP4 differences.
Until independently calibrated FP32 checks justify qualification, native/custom TP quality uses EXPLORATORY status.
Exploratory generation requires finite logits, valid vocabulary IDs, audited weights, and live greedy feedback.
Quality and numerical status remain independent. Task accuracy never retroactively passes the old gate.
No exploratory timing constitutes qualified architecture speedup.
''')
    (ROOT/'eval_protocol.md').write_text('''# Frozen evaluation protocol

Evaluate the same 64 GSM8K and 50 MATH-500 pilot IDs for every arm.
Use the pinned base Ouro checkpoint, tokenizer, prompt strings, greedy decoding, and 4096-token cap.
Allow native EOS. Preserve the frozen per-question stop sequences.
Store every actual generated token, prompt checksum, answer, parse status, and stop reason.
Report answer accuracy and completed-answer accuracy separately.
Use the existing GSM numeric parser and Math-Verify symbolic grader without candidate-specific extraction.
Bootstrap paired question IDs, not tokens or layers, with 5000 draws and seed 20261021.
Full evaluation is 1319/500. If resources prevent it, show NOT_RUN, not extrapolated pilot accuracy.
Speedup requires the same completed question set. Report differing output/depth work explicitly.
Controlled replay remains separate from actual answer generation.
''')
    print('FROZEN',len(ordered),'questions; 114 pilot requests; 8 arms',flush=True)

if __name__=='__main__':main()
