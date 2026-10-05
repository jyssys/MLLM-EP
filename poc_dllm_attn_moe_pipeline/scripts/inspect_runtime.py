"""Freeze metadata-only fresh split before observing routing/timing."""
import json
import random
import subprocess
from datetime import datetime, timezone
from poc_dllm_attn_moe_pipeline.contract import *


def main():
    excluded = set()
    for directory in ['poc_dllm_ep_comm', 'poc_dllm_request_persistent_replica']:
        rows = json.loads((PROJECT / directory / 'results/frozen_manifest.json').read_text())['rows']
        excluded.update(x.get('task', x['prompt']) for x in rows)
    source = Path('/home/esjung/MLLM-EP-llada2-flash-100b/poc_llada2_flash_ep/data')
    sources = {'coding': source/'bounded_eval_100/humaneval_100.json',
               'reasoning': source/'bounded_eval_100/gsm8k_100.json'}
    rows = []
    for domain, path in sources.items():
        items = json.loads(path.read_text())['details']
        indices = [i for i, x in enumerate(items) if x['prompt'] not in excluded]
        random.Random(10062026).shuffle(indices)
        assert len(indices) >= 16
        for n, i in enumerate(indices[:16]):
            rows.append(dict(request_id=f'{domain}_{n:02d}', problem_id=f'{domain}:{i}',
                             domain=domain, source_index=i, prompt=items[i]['prompt'],
                             split='calibration' if n < 4 else 'heldout',
                             early_kill=n < (2 if n < 4 else 8), generation_budget=64))
    rows.sort(key=lambda x: (x['split'] != 'calibration', int(x['request_id'].split('_')[1]), x['domain']))
    result = ROOT/'results/main'
    if (result/'design_freeze.json').exists():
        raise FileExistsError('Do not overwrite frozen selection')
    write(result/'design_freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        selection='seed10062026 metadata-only, excludes earlier amplification and residency task identities',
        rows=rows, early_kill='4 calibration + 8 heldout; remaining20 only after positive seam',
        native=dict(B=1, block=32, threshold=.9, temperature=0, gen_length=64, EOS=True),
        boundaries=[1,9,17], engineered_groups=2, numerical_tolerances=dict(moe_max_abs=.03125,
        moe_relative_l2=.002, attention_max_abs=.03125, attention_relative_l2=.003),
        promotion=dict(envelope_kill=.05, envelope_support=.10, seam_gain=.10,
                       request_projection=.05, full_gain=.07, generic_increment=.03),
        statistics='2000 request-cluster paired bootstrap; no layer/iteration pseudo-replication',
        sources={k:dict(path=str(p), sha256=sha(p)) for k,p in sources.items()},
        candidate='none until dependency and split-cost gates', sources_previous_observed=True))
    (ROOT/'references').mkdir(exist_ok=True)
    (ROOT/'references/authoritative_spec.md').write_bytes(SPEC.read_bytes())
    command = lambda x: subprocess.check_output(x, text=True)
    write(result/'environment.json', dict(spec_sha256=sha(SPEC), mask=MASK,
        logical_physical=dict(enumerate(GPUS)), uuids=UUIDS, model_revision=REVISION,
        model_config=json.loads((MODEL/'config.json').read_text()),
        gpus=command(['nvidia-smi','--id='+MASK,'--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),
        topology=command(['nvidia-smi','topo','-m']),
        git_parent=command(['git','-C',str(PROJECT),'rev-parse','HEAD']).strip(),
        dinfer_sha=command(['git','-C',str(DINFER),'rev-parse','HEAD']).strip(),
        source_sha256=sha(DINFER/'python/dinfer/model/modeling_llada2_moe_sglang.py'),
        source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(DINFER),'diff','--binary'])).hexdigest(),
        prior_worker_sha256=sha(PROJECT/'poc_dllm_ep_comm/scripts/run_dllm_ep4.py'),
        initial_burner_state='NONE; authoritative contract sec6 restores only pre-existing state, no new unmasked service'))
    print('FROZEN32 / EARLY_KILL12', flush=True)


if __name__ == '__main__':
    main()
