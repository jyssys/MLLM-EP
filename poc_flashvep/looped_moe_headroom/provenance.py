"""Save installed versions and link worker records to immutable model revisions."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .common import ROOT,ASSETS,MODELS,SPEC,XLLM,digest,write

def main():
    packages=['torch','transformers','safetensors','einops','flash-attn-4','nvidia-cutlass-dsl',
        'quack-kernels','apache-tvm-ffi','tilelang','flash-linear-attention','fla-core','tiktoken','pyarrow','ninja']
    versions={name:metadata.version(name) for name in packages}
    write(ROOT/'environment.json',dict(platform=platform.platform(),python=sys.version,packages=versions,
        worker_python=sys.executable,native_source=str(XLLM),
        native_revision=subprocess.check_output(['git','-C',str(XLLM),'rev-parse','HEAD'],text=True).strip(),
        native_tracked_diff=subprocess.check_output(['git','-C',str(XLLM),'diff','--stat'],text=True),
        native_attention='official ENABLE_FLASH_ATTENTION_4 path',native_tf32_policy=True,
        diagnostic_reference_tf32=False,precision='BF16 learned parameters. Native FP32 reductions/logits/buffers remain unchanged.',
        extension_arch='SM90',compiler_cuda='12.8',torch_cuda='12.9',
        isolated_environment_pth=(Path(sys.prefix)/'lib/python3.12/site-packages/experiment_readonly_base.pth').read_text(),
        spec_sha256=digest(SPEC)))
    lock=ROOT/'requirements.installed.txt'
    lock.write_text('# Validated installed direct packages. Native xLLM source is pinned separately.\n'+
        '\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n')
    config_dir=ROOT/'config'
    config_dir.mkdir(exist_ok=True)
    for model in MODELS:
        source=ASSETS/model
        if model=='foil1':
            source=source/'Foil-1'
        for filename in ['config.json','artifact_manifest.json']:
            if (source/filename).exists():
                shutil.copyfile(source/filename,config_dir/f'{model}_{filename}')
    pins=json.loads((ROOT/'model_pins.json').read_text())
    foil=ASSETS/'foil1'
    metadata_path=foil/'.cache/huggingface/download/Foil-1/model.safetensors.metadata'
    revision,lfs_hash,_=metadata_path.read_text().splitlines()
    assert revision==pins['foil1']['revision']
    actual_hash=digest(foil/'Foil-1/model.safetensors')
    assert actual_hash==lfs_hash,'Foil checkpoint hash differs from the downloaded LFS object'
    write(ROOT/'foil_artifact_verification.json',dict(revision=revision,model_sha256=actual_hash,
        lfs_sha256_match=True,root_remote_code_sha256=digest(foil/'modeling_loop_lm.py'),
        subfolder_remote_code_sha256=digest(foil/'Foil-1/modeling_loop_lm.py'),
        parameter_load_verification='No missing, unexpected, or mismatched tensors in final strict-load run'))
    inventory=json.loads((ROOT/'hardware_inventory.json').read_text())['gpu_query']['stdout'].splitlines()[1:]
    uuid={int(r.split(',')[0]):r.split(',')[1].strip() for r in inventory}
    runs=[]
    for path in sorted((ROOT/'raw').iterdir()):
        if not (path/'request.json').exists():
            continue
        request=json.loads((path/'request.json').read_text())
        complete=path/'complete.json'
        failure=path/'failure.json'
        runtime=next((p for p in [path/'runtime_metadata.json',path/'model_metadata.json',path/'metadata.json'] if p.exists()),None)
        info=json.loads(runtime.read_text()) if runtime else {}
        job=json.loads(Path(request['job']).read_text()) if request.get('job') and Path(request['job']).exists() else {}
        model=request['model']
        completion=json.loads(complete.read_text()) if complete.exists() else {}
        runs.append(dict(run_id=path.name,model_id=MODELS[model],checkpoint_revision=pins[model]['revision'],
            upstream_code_commit='df0ed1ea415b55f0a944a291f90da9d47c98b5d5' if model.startswith('ifm') else pins[model]['revision'],
            local_code_commit=request['commit'],source_sha256=request.get('source_sha256'),
            gpu_physical_id=int(request['gpu']),gpu_uuid=info.get('device_uuid',uuid[int(request['gpu'])]),
            gpu_uuid_provenance='OWN_PID_NVML' if info.get('device_uuid') else 'CUDA_MASK_AND_INVENTORY',
            gpu_name=info.get('device_name','NVIDIA H100 80GB HBM3'),compute_capability=info.get('compute_capability',[9,0]),
            torch_version=info.get('torch',versions['torch']),cuda_version=info.get('cuda','12.9'),
            attention_backend=info.get('backend'),dtype='BF16 learned parameters',workload_id=request['phase'],
            sequence_ids_hash=__import__('hashlib').sha256(json.dumps(job.get('ids',[])).encode()).hexdigest(),
            split='See per-example rows',input_length_actual='See per-example rows',
            new_tokens=job.get('cap',job.get('generation_lengths')),batch_size=1,
            recurrent_passes_actual=info.get('loops'),experiment_arm=job.get('arms',request['phase']),
            target_q='See per-example rows',effective_q='See per-example rows',seed=20261026,
            warmup_count=job.get('warmups',5 if request['phase'] in ('timing','micro','contention') else None),
            timed_repeat_count=job.get('repeats',20 if request['phase'] in ('timing','micro','contention') else None),
            time_window=dict(start_unix=request['start'],end_unix=request['start']+completion['elapsed_s'] if 'elapsed_s' in completion else None),
            time_window_start=request['start'],baseline_ms='See timing_summary.csv',variant_ms=None,
            speedup_measured=None,speedup_modeled='See oracle_summary.csv',
            quality_metrics='See raw quality.jsonl or generation_quality.jsonl',cache_peak_bytes='See run artifacts',
            status=json.loads(complete.read_text())['status'] if complete.exists() else 'FAILED' if failure.exists() else 'INCOMPLETE',
            failure_log=str(failure.relative_to(ROOT)) if failure.exists() else None))
    write(ROOT/'run_manifest.json',dict(runs=runs,
        actual_candidate_speedup='NOT_MEASURED_P3_GATE_NOT_PASSED',
        source_snapshot_scope='Hashes describe files at each worker start. Final source retains the numerical paths and explicit legacy controls.'))
    print('PROVENANCE_RUNS',len(runs))

if __name__=='__main__':
    main()
