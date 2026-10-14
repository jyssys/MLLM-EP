"""Read-only audit and metadata freeze. Never replace frozen records."""
import datetime
import importlib.metadata
import shutil
import subprocess
from poc_dllm_multigpu_e2e_atlas.contract import *


def main():
    target=ROOT/'manifests/environment.json'
    assert not target.exists(),'Preserve the initial audit.'
    def cmd(args,cwd=PROJECT):return subprocess.check_output(args,cwd=cwd,text=True)
    shutil.copyfile(SPEC,ROOT/'authoritative_spec.md')
    original=SPEC.parents[2]
    sources=['model/modeling_llada2_moe_sglang.py','decoding/diffusion_runner.py',
        'decoding/generate_uniform.py','decoding/utils.py','decoding/parallel_strategy.py']
    write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent=cmd(['git','rev-parse','HEAD']).strip(),mask=MASK,physical_GPUS=GPUS,UUIDS=UUIDS,
        logical_to_physical=dict(enumerate(GPUS)),spec_sha256=sha(SPEC),
        nvidia_smi=cmd(['nvidia-smi']),topology=cmd(['nvidia-smi','topo','-m']),
        nsys=cmd(['nsys','--version']).strip(),
        versions={name:importlib.metadata.version(name) for name in ['torch','sglang','deep-ep','triton','transformers']},
        dinfer_revision=cmd(['git','rev-parse','HEAD'],DINFER).strip(),
        dinfer_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'],cwd=DINFER)).hexdigest(),
        source_hashes={s:sha(DINFER/'python/dinfer'/s) for s in sources},
        original_staged_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--cached'],cwd=original)).hexdigest(),
        original_unstaged_sha256=hashlib.sha256(subprocess.check_output(['git','diff'],cwd=original)).hexdigest()))
    config=json.loads((MODEL/'config.json').read_text())
    write(ROOT/'manifests/model_geometry.json',dict(model='inclusionAI/LLaDA2.0-mini',revision=REVISION,
        config=config,config_sha256=sha(MODEL/'config.json'),precision='BF16',native='TP4/EP1/DP1/PP1/CP1',
        EP_dispatch_all_to_all='NOT_PRESENT',EP_combine_all_to_all='NOT_PRESENT',
        actual_collectives='Verify capture metadata and physical profile, not labels alone'))
    old=PROJECT/'poc_dllm_prefix_canvas_cp/results/frozen_manifest.json'
    frozen=json.loads(old.read_text())
    for split in ['calibration','held_out']:
        write(ROOT/f'manifests/workload_{split}.json',dict(
            source=str(old.relative_to(PROJECT)),source_sha256=sha(old),
            token_artifact='poc_dllm_prefix_canvas_cp/results/frozen_tokens.json.gz',
            token_sha256=sha(PROJECT/'poc_dllm_prefix_canvas_cp/results/frozen_tokens.json.gz'),
            requests=[r for r in frozen['requests'] if r['split']==split],
            prefixes=PREFIXES,batch=1,block=BLOCK,output=OUTPUT,threshold=.9,temperature=0,seed=SEED,
            new_heldout_claim=False,selection='Reuse frozen metadata. Prior calibration is continuity, not fresh confirmation.',
            repeated_measures='3 clean and3 coarse/request. Additional parity-only audit is not headline timing.',
            promotion=dict(removable_E2E_fraction=.15,realistic_2x_E2E_gain=.10,
                observer_preferred=.05,observer_reject=.10),
            chronology='No candidate before new Stage A and architecture screening.'))
    print('Frozen existing8 calibration and24 untouched held-out IDs. Native TP4, no EP A2A.')


if __name__=='__main__':main()
