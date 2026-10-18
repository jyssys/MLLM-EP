"""Freeze metadata and retrieve the immutable base checkpoint before inference."""
import datetime
import hashlib
import os
import subprocess
from huggingface_hub import snapshot_download
from poc_decode_first_depth_aware_dp.contract import *


def main():
    directory=ROOT/'artifacts';directory.mkdir(exist_ok=False)
    (ROOT/'reports').mkdir(exist_ok=True)
    # Preserve all previous specs. The original project's reports/spec.md is untouched.
    (ROOT/'reports/spec.md').write_bytes(SPEC.read_bytes())
    assert sha(ROOT/'reports/spec.md')==sha(SPEC)
    parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    previous=read(PROJECT/'poc_async_recurrent_pp/outputs/manifests/data.json')
    rank=lambda row:hashlib.sha256(('decode-first-v2-20261018/'+row['id']).encode()).hexdigest()
    ordered={d:sorted([r for r in previous['full'] if r['dataset']==d],key=rank) for d in ['gsm8k','math500']}
    calibration=previous['calibration'][:3]
    assert all('/train/' in r['id'] for r in calibration)
    demonstrations='\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer'] for r in calibration)
    def prompt(row):
        suffix="\nAnswer: Let's think step by step. Put the final answer in \\boxed{...}."
        return (demonstrations+'\n\n' if row['dataset']=='gsm8k' else '')+'Question: '+row['question']+suffix
    rows=[]
    for d in ordered:
        for i,row in enumerate(ordered[d]):
            rows.append(dict(**row,prompt=prompt(row),split='pilot' if i<16 else 'heldout',order=i))
    write(directory/'frozen_workload.json',dict(rows=rows,source='Previous immutable full benchmark manifest. New metadata-only hash order.',
        gsm_revision=previous['gsm_revision'],math_revision=previous['math_revision'],
        prompt_style='GSM8K three fixed training examples. MATH-500 zero-shot. No chat template for base model.',
        demonstration_ids=[r['id'] for r in calibration],pilot_per_domain=16,full_counts={d:len(ordered[d]) for d in ordered},
        quality_generation=dict(temperature=0,max_new_tokens=4096,truncation_followup_cap=8192,EOS_allowed=True),
        prompt_sha256={r['id']:hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows}))
    original=Path('/home/esjung/MLLM-EP-github')
    dirty={label:hashlib.sha256(subprocess.check_output(command,cwd=original)).hexdigest()
        for label,command in [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]}
    write(directory/'preregistration.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=parent,
        spec_source=str(SPEC),spec_sha256=sha(SPEC),original_tracked_diff_sha256=dirty,
        primary_checkpoint=MODEL_ID,revision=REVISION,checkpoint_switch='Base candidate replaces unqualified Thinking adaptive gate. No accuracy transfer.',
        runtime='Official continuous-depth-batching research implementation',CDB_revision=CDB_REVISION,RLT_audit_revision=RLT_REVISION,
        primary_gate=dict(exit_threshold=.5,min_recurrent_steps=2,max_recurrent_steps=4,
            type='Original same-step cumulative hazard. No trained replacement gate.',delay_gate_consumption=False),
        KV_policy='last_exited: independent depth slots with exit-KV propagation; no single shared slot',
        quality_guardrail_pp=dict(gsm8k=1.,math500=2.),
        severe_pilot_stop='At least 15 pp accuracy loss within either domain and paired bootstrap lower loss bound above zero. No threshold rescue.',
        further_rule='No scheduling claims before actual skipped execution and quality gates. Uncertain pilot expands; full quality not inferred from smoke.',
        numerical_limits=dict(logits_max_abs=.5,logits_mean_abs=.05,KV_relative_L2=.02),
        replicas='DP4 independent TP1/EP1/PP1/CP1, physical GPUs 4–7',RL=False,
        oracle_objective='P95 E2E, with throughput and fairness retained. No post-prefill migration.',
        calibration_policy='No gate tuning on heldout rows. No fabricated depth replay as online execution.'))
    result=snapshot_download(MODEL_ID,revision=REVISION,allow_patterns=['*.json','*.py','*.safetensors','merges.txt','vocab.json','README.md'],max_workers=4)
    assert Path(result)==MODEL
    for repo,revision in [(CDB,CDB_REVISION),(RLT,RLT_REVISION)]:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==revision
    write(directory/'checkpoint.json',dict(model_id=MODEL_ID,revision=REVISION,path=str(MODEL),
        hashes={p.name:sha(p) for p in MODEL.iterdir() if p.is_file() and p.name!='README.md'},config=read(MODEL/'config.json')))
    print('PREPARED',MODEL_ID,REVISION,len(rows),'quality rows',flush=True)


if __name__=='__main__':main()
