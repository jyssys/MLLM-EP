"""Bounded, sequential controls for the resumed baseline screen.

Launch only after the main clean screen releases all four GPUs. These are
reference-port and length/membership diagnostics, not successor mechanisms.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices

p = argparse.ArgumentParser()
p.add_argument('--result', type=Path, required=True)
p.add_argument('--only', default='metadata,scheduler,fixed1,fixed16')
a = p.parse_args()
allowed_devices()
r = a.result.resolve()
task = Path(__file__).resolve().parent
py = '/home/esjung/.venvs/flashvep-deepep-v020/bin/python'
cpu = '/home/esjung/.venvs/successor-research-audit/bin/python'
model = '/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c'
env = dict(os.environ, OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    SUCCESSOR_SERE_EXTENSION='/home/esjung/.cache/torch_extensions/py312_cu129/sere_successor_reference/sere_successor_reference.so')
original = json.loads((r/'analysis/combined_confirmatory_policies_20260908.json').read_text())
cached = [x if x['method']=='vanilla' else dict(x, name=x['name']+'_metadata_cached', cache_token_metadata=True)
          for x in original]
(r/'analysis/cached_confirmatory_policies_20260908.json').write_text(json.dumps(cached, indent=2))
diagnostic = [x for x in cached if x['name'] in ('vanilla',
    'sere_s2_r05_official_metadata_cached', 'modes_official_target70_metadata_cached')]
(r/'analysis/route_diagnostic_policies_20260908.json').write_text(json.dumps(diagnostic, indent=2))
phase_control = [x for x in cached if x['method'] in ('vanilla','modes')]
phase_control += [dict(x, name=x['name']+'_prefill_only', phase='prefill')
                  for x in cached if x['method']=='modes']
(r/'analysis/modes_phase_control_policies_20260908.json').write_text(json.dumps(phase_control, indent=2))
decision_control = [x for x in cached if x['name'] in ('vanilla',
    'sere_s2_r05_official_metadata_cached','sere_s4_r05_official_metadata_cached',
    'modes_official_target70_metadata_cached')]
decision_control += [dict(next(x for x in cached if x['method']=='sere'),
                         name='sere_noop_s8_metadata_cached',retain=8),
                     dict(next(x for x in cached if x['method']=='modes'),
                         name='modes_noop_metadata_cached',tau_text=0.0,tau_vision=0.0)]
(r/'analysis/decision_cost_control_policies_20260908.json').write_text(json.dumps(decision_control, indent=2))
conditions = {
    'metadata': (1,32,3,'port_metadata_control_policies_20260908.json',[]),
    'scheduler': (16,32,1,'route_diagnostic_policies_20260908.json',
                  ['--scheduler-context','--routes','--instrument']),
    'scheduler1': (1,16,1,'route_diagnostic_policies_20260908.json',
                  ['--scheduler-context','--routes','--instrument']),
    'fixed1': (1,16,3,'cached_confirmatory_policies_20260908.json',['--fixed-output']),
    'fixed16': (16,32,3,'cached_confirmatory_policies_20260908.json',['--fixed-output']),
    'phase': (1,64,3,'modes_phase_control_policies_20260908.json',[]),
    'decision': (16,32,3,'decision_cost_control_policies_20260908.json',['--fixed-output']),
}
for index, key in enumerate(a.only.split(',')):
    batch, count, reps, catalog, flags = conditions[key]
    name = 'ep_port_control_'+key+'_20260908'
    out = r/'online'/name
    assert not out.exists(), out
    command = [py,str(task/'run_ep_serving.py'),'--model',model,
        '--data',str(r/'data/requests.jsonl'),'--policies',str(r/'analysis'/catalog),
        '--out',str(out),'--dataset','chartqa','--requests',str(count),
        '--batch-per-dp',str(batch),'--output-tokens','32','--warmups','2',
        '--repetitions',str(reps),'--order-seed',str(9321+index),*flags]
    started = datetime.now(timezone.utc).isoformat()
    print(json.dumps(dict(name=name,started_utc=started,command=command)),flush=True)
    with (r/'raw'/f'{name}.log').open('w') as log:
        result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    with (r/'raw/resume_control_runs_20260908.jsonl').open('a') as log:
        log.write(json.dumps(dict(name=name,started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode))+'\n')
    if result.returncode: raise SystemExit(result.returncode)
    with (r/'raw'/f'{name}_analysis.log').open('w') as log:
        subprocess.run([cpu,str(task/'analyze_ep_serving.py'),'--input',str(out),
            '--output',str(r/'analysis'/name),'--official-source',
            str(r/'references/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py')],
            check=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(completed=name)),flush=True)
