"""Freeze long prompts, split, native decoding and policy gates before capture."""
import json
import random
import subprocess
from datetime import datetime,timezone
from poc_dllm_request_persistent_replica.contract import *

CODE_TEMPLATE='''Write a complete, self-contained Python module for the following public programming task. Include the requested implementation with clear docstrings, a small command-line demonstration, and a unittest test class covering ordinary cases and edge cases. Keep the tests consistent with the task. Output only the complete Python code, not a summary or prose explanation.\n\nTask specification and starter signature:\n{task}'''
REASON_TEMPLATE='''Solve the following mathematics problem. Explain the calculation step by step, include a separate verification of the result and why a plausible alternative calculation is wrong, and end with the numerical answer.\n\n{task}'''


def main():
    original=json.loads((COMM/'results/frozen_manifest.json').read_text())['rows']
    seen={x['prompt'] for x in original}
    source=Path('/home/esjung/MLLM-EP-llada2-flash-100b/poc_llada2_flash_ep/data')
    paths={'coding':source/'bounded_eval_100/humaneval_100.json',
           'reasoning':source/'bounded_eval_32/gsm8k_32.json'}
    rows=[];sources={}
    for domain,count,cal,template in [('coding',24,6,CODE_TEMPLATE),('reasoning',8,2,REASON_TEMPLATE)]:
        file=paths[domain];items=json.loads(file.read_text())['details']
        order=[i for i,x in enumerate(items) if x['prompt'] not in seen]
        random.Random(9017).shuffle(order)
        assert len(order)>=count
        sources[domain]=dict(path=str(file),sha256=sha(file),origin='public HumanEval task' if domain=='coding' else 'public GSM8K task')
        for n,i in enumerate(order[:count]):
            rows.append(dict(request_id=f'{domain}_{n:02d}',domain=domain,source_index=i,
                prompt=template.format(task=items[i]['prompt']),task=items[i]['prompt'],
                split='calibration' if n<cal else 'heldout',generation_budget=512))
    destination=ROOT/'results/frozen_manifest.json'
    new_json(destination,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),rows=rows,sources=sources,
        source_selection='seed9017 metadata permutation, excluding previous amplification prompts; no generated-output/routing selection',
        primary='long-form self-contained coding module+unit-tests, B1 communication-exposed regime; reasoning secondary',
        templates=dict(coding=CODE_TEMPLATE,reasoning=REASON_TEMPLATE),
        model='inclusionAI/LLaDA2.0-mini',revision=REVISION,dtype='BF16',
        native=dict(block_length=32,threshold=.9,temperature=0,generation_budget=512,EOS_enabled=True,
                    cache='prefix',maximum_unroll=1,expected_tpf=15,early_stop=True),
        budgets_gib=BUDGETS,load_caps=[.10,.15,.20],
        policies=['global_frequency','global_destination','first_block','first2_blocks','oracle','oracle_load'],
        copy='once per request after full observed block(s); global static prewarmed free; no per-block reload or optional incremental update',
        load_guard='causal guards use observed blocks ONLY; global guard calibration ONLY; future-truth load guard oracle ONLY; evaluate actual future worst-stage load, never prune using heldout truth',
        byte_scope='actual masked output rows before forward only; exclude context/alreadyaccepted; original acceptance-route denominator fixed',
        memory='shared layer-specific capacity across ALL19 sparse layers per physical source rank',
        planning='reuse frozen destination-bundle greedy family; global destination planner stronger control beside popularity',
        primary_selection='calibration only: largest relative gain over stronger global at +10% guard and <=2GiB; heldout untouched for selection',
        gates=dict(relative_gain=.10,request_bootstrap_ci_low=0.,future_blocks=4,copy_roi=2.,break_even_before=.50,
                   worst_stage_load=.10,oracle_recovery_beyond_global=.50),
        statistics='2000 request bootstrap; block-distance records clustered by request',
        execution='32longrequests x bypass/timing/trace plus3tinywarmups; per-request300s / session3600s watchdog; no unbounded retries',
        no_replication_runtime_before_stage_a=True))
    (ROOT/'references').mkdir(exist_ok=True)
    (ROOT/'references/authoritative_spec.md').write_bytes(SPEC.read_bytes())
    command=lambda x:subprocess.check_output(x,text=True)
    new_json(ROOT/'results/environment.json',dict(parent=PARENT,model_revision=REVISION,
        model_config=json.loads((MODEL/'config.json').read_text()),dinfer_sha=command(['git','-C',str(DINFER),'rev-parse','HEAD']).strip(),
        previous_environment=json.loads((SHORT/'results/environment.json').read_text()),
        original_spec_sha256=sha(SPEC),gpu_mask=MASK,logical_physical=dict(enumerate(GPUS)),
        gpus=command(['nvidia-smi','--id='+MASK,'--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),
        topology=command(['nvidia-smi','topo','-m']),
        borrowed_p2p= dict(path='poc_dllm_block_local_replica/raw/physical_revalidation/p2p/p2p.json',
                          sha256=sha(SHORT/'raw/physical_revalidation/p2p/p2p.json')),
        validated_worker_sha256=sha(COMM/'scripts/run_dllm_ep4.py'),source_no_changes=True))
    print('FROZEN_LONG_COHORT',len(rows),'calibration8 heldout24, genbudget512',flush=True)


if __name__=='__main__':main()
