"""Freeze provenance and runtime limits without allocating GPUs."""
import datetime
import importlib.metadata
import shutil
import subprocess
from poc_dllm_prefix_canvas_cp.contract import *


def main():
    def cmd(args,cwd=PROJECT):return subprocess.check_output(args,cwd=cwd,text=True)
    results=ROOT/'results';results.mkdir(parents=True,exist_ok=True)
    assert not (results/'environment.json').exists(),'Preserve the frozen initial audit. Use a new experiment directory.'
    assert not (results/'model_runtime_contract.json').exists(),'Preserve the frozen model audit.'
    snapshot=ROOT/'authoritative_spec.md'
    if not snapshot.exists():shutil.copyfile(SPEC,snapshot)
    assert sha(snapshot)==sha(SPEC)
    source=DINFER/'python/dinfer'
    paths=['model/modeling_llada2_moe_sglang.py','decoding/utils.py','decoding/generate_uniform.py','decoding/diffusion_runner.py']
    write(results/'environment.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        physical_GPUS=GPUS,UUIDS=UUIDS,physical_to_logical=dict(zip(GPUS,range(4))),
        inventory=cmd(['nvidia-smi']),topology=cmd(['nvidia-smi','topo','-m']),
        git_parent=cmd(['git','rev-parse','HEAD']).strip(),spec_sha256=sha(SPEC),
        dinfer_revision=cmd(['git','rev-parse','HEAD'],DINFER).strip(),
        dinfer_dirty_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff'],cwd=DINFER)).hexdigest(),
        dinfer_source_hashes={p:sha(source/p) for p in paths},
        versions={p:importlib.metadata.version(p) for p in ['torch','sglang','transformers','deep-ep','triton']},
        original_staged_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff','--cached'],cwd=SPEC.parents[2])).hexdigest(),
        original_unstaged_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff'],cwd=SPEC.parents[2])).hexdigest()))
    cfg=json.loads((MODEL/'config.json').read_text())
    assert cfg['max_position_embeddings']==MAX_CONTEXT and not cfg['use_sliding_window']
    write(results/'model_runtime_contract.json',dict(model='inclusionAI/LLaDA2.0-mini',revision=REVISION,
        preview=False,config=cfg,config_sha256=sha(MODEL/'config.json'),
        old_runtime_max_context=4096,configured_max_context=MAX_CONTEXT,
        configured_context_status='MODEL_SUPPORTED; physical allocator/long-offset probe pending',
        native_attention='torch SDPA auto-selected optimized backend; backend verified in separate profiler lane',
        primary_layout='native TP4 / EP1 / DP1 / PP1; head-sharded KV, no CP',
        reason='Native benchmark defaults to no A2A; attention study must not inherit avoidable source-SP EP bridge overhead',
        primary_prefilling_limit=MAX_CONTEXT,primary_output_budget=OUTPUT,block=BLOCK,threshold=.9,temperature=0,
        graph='native no-A2A graph support audited; initial eager smoke, bounded graph validation before profiling',
        supported_prefixes=PREFIXES,skipped_prefixes={'32768':'P + output64 exceeds trained context32768'},
        tolerance_frozen={'observer_mean_preferred':.05,'observer_upper_CI_ceiling':.10,
            'token_acceptance_NFE':'EXACT same-config observer parity'},
        preliminary_mask_warning='Native SGLang cache includes padded zero slots. Preserve and audit the actual mask; do not silently repair it.'))
    print('AUDIT',REVISION,'trained context',MAX_CONTEXT,'native TP4 preferred')


if __name__=='__main__':main()
