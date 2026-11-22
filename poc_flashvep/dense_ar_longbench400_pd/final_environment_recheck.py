"""Verify frozen model and installed Python sources after GPU work drains."""
import importlib.metadata
import hashlib
import json
from pathlib import Path
import time
import zipfile
from .runtime_binding import activate
activate()
from .common import ROOT,REPO,SOURCE,write,digest,preserve


def main():
    assert all((ROOT/'state'/f'final_performance_lane{i}_complete_v1.json').exists() for i in range(4))
    assert (ROOT/'state/controlled_continuous_bridge_complete_v1.json').exists()
    assert (ROOT/'state/hybrid_READ_repair_complete_v1.json').exists()
    assert (ROOT/'state/hybrid_READ_repair_layer_control_complete_v1.json').exists()
    assert (ROOT/'state/request_registration_deferral_complete_v1.json').exists()
    assert (ROOT/'state/request_registration_q28_extension_complete_v1.json').exists()
    assert (ROOT/'state/request_registration_controlled_bridge_complete_v1.json').exists()
    for version in (1,2,3):
        for axis in ('Layer','Hybrid'):
            assert (ROOT/'state'/f'Decode_compile_{axis}_gate_v{version}.json').exists()
    for axis in ('Layer','Hybrid'):
        assert (ROOT/'state'/f'Decode_IR_{axis}_gate_v1.json').exists()
        assert (ROOT/'state'/f'Scoped_compile_{axis}_gate_v1.json').exists()
    assert (ROOT/'state/scoped_compile_teachers_complete_v1.json').exists()
    assert (ROOT/'state/scoped_compile_confirmation_complete_v1.json').exists()
    assert (ROOT/'state/eager_compile_confirmation_v2_complete_v1.json').exists()
    assert (ROOT/'state/eager_compile_teachers_complete_v2.json').exists()
    for axis in ('Layer','Hybrid'):
        assert (ROOT/'state'/f'Eager_decode_{axis}_gate_v2.json').exists()
    preserve()
    from .quality_dependency_audit import main as quality_dependencies
    quality_dependencies()
    model=json.loads((ROOT/'references/model_tokenizer_file_hashes_v1.json').read_text())
    model_results=[]
    for row in model['files']:
        path=Path(model['snapshot'])/row['file']
        assert path.resolve()==Path(row['resolved_path']).resolve(),path
        assert path.stat().st_size==row['bytes'] and digest(path)==row['SHA256'],path
        model_results.append(dict(file=row['file'],bytes=row['bytes'],SHA256=row['SHA256'],unchanged=True))
    runtime=json.loads((ROOT/'references/installed_runtime_source_audit_v1.json').read_text())
    assert digest(ROOT/'references/installed_vllm_python_sources_v1.zip')==runtime['source_archive_SHA256']
    licenses=json.loads((ROOT/'references/installed_vllm_source_licenses_metadata_v1.json').read_text())
    assert digest(ROOT/'references/installed_vllm_source_licenses_metadata_v1.zip')==licenses['archive_SHA256']
    import vllm
    parent=Path(vllm.__file__).resolve().parent.parent
    for row in runtime['files']:
        assert digest(parent/row['path'])==row['SHA256'],row['path']
    versions={name:importlib.metadata.version(name) for name in runtime['versions']}
    assert versions==runtime['versions'],versions
    plan=json.loads((ROOT/'configs/final_continuous_execution_plan_v1.json').read_text())
    capsule=ROOT/'references/source_capsules'/plan['source_capsule']
    assert digest(capsule)==plan['source_capsule_SHA256']
    inference_files=('adapter.py','stock_token_adapter.py','compiled_full_adapter.py','serving.py',
        'standard_worker.py','hidden_transport.py','decode_prefill_throttle.py','remote_setup_deferral.py',
        'prefill_completion_guard.py','graph_guard.py','chunk_boundary.py','geometry.py','runtime_binding.py')
    actor_hashes=[]
    with zipfile.ZipFile(capsule) as archive:
        for name in inference_files:
            path=SOURCE/name
            sha=hashlib.sha256(archive.read(str(path.relative_to(REPO)))).hexdigest()
            assert digest(path)==sha,path
            actor_hashes.append(dict(file=name,SHA256=sha,unchanged=True))
    registration=json.loads((ROOT/'configs/request_registration_deferral_frozen_v1.json').read_text())
    overlay=ROOT/'references/source_capsules'/registration['source_capsule']
    meta=json.loads(overlay.with_suffix('.json').read_text())
    assert digest(overlay)==registration['source_capsule_SHA256']==meta['SHA256']
    assert meta['base_capsule_SHA256']==plan['source_capsule_SHA256']
    with zipfile.ZipFile(capsule) as original,zipfile.ZipFile(overlay) as added:
        changed=[n for n in original.namelist() if original.read(n)!=added.read(n)]
        assert changed==[meta['only_changed_existing_file']]
        assert set(added.namelist())-set(original.namelist())=={meta['added_file']}
        assert added.read(meta['added_file'])==(SOURCE/'request_admission_deferral.py').read_bytes()
    compiler_overlays=[]
    for name in ('decode_only_compile_overlay_v1.zip','decode_only_compile_custom_ops_overlay_v1.zip',
                 'decode_IR_priority_overlay_v1.zip','scoped_decode_compile_overlay_v1.zip',
                 'eager_prefill_native_decode_overlay_v1.zip','eager_prefill_native_decode_overlay_v2.zip'):
        path=ROOT/'references/source_capsules'/name
        entry=json.loads(path.with_suffix('.json').read_text())
        base=ROOT/'references/source_capsules'/entry['base_capsule']
        assert digest(path)==entry['SHA256'] and digest(base)==entry['base_capsule_SHA256']
        with zipfile.ZipFile(base) as original,zipfile.ZipFile(path) as added:
            actual_changed=sorted(n for n in original.namelist() if original.read(n)!=added.read(n))
            actual_added=sorted(set(added.namelist())-set(original.namelist()))
            assert actual_changed==sorted(entry['changed_existing_files'])
            assert actual_added==sorted(entry['added_files'])
            assert set(entry['files'])==set(added.namelist())
            assert all(hashlib.sha256(added.read(n)).hexdigest()==sha for n,sha in entry['files'].items())
        compiler_overlays.append(dict(name=name,SHA256=entry['SHA256'],
            base_SHA256=entry['base_capsule_SHA256'],changed=actual_changed,added=actual_added))
    preserve()
    write(ROOT/'state/final_model_runtime_integrity_recheck.json',dict(time=time.time(),passed=True,
        model_revision=model['revision'],model_files=model_results,versions=versions,
        installed_python_files_verified=len(runtime['files']),installed_python_sources_unchanged=True,
        final_inference_source_SHA256=plan['source_capsule_SHA256'],inference_files=actor_hashes,
        adaptive_registration_source_SHA256=registration['source_capsule_SHA256'],
        adaptive_overlay_only_changed=changed,adaptive_overlay_added=meta['added_file'],
        role_local_compiler_overlays=compiler_overlays,
        initial_runtime_audit_SHA256=digest(ROOT/'references/installed_runtime_source_audit_v1.json'),
        scope='End-of-GPU-work checks reproduce the initial model/tokenizer file and installed vLLM Python source hashes. '
            'This does not prove that the installed wheel is a canonical upstream wheel or independently verify every GPU binary. '
            'No runtime,model,user checkout or staged-index file is changed.'))
    print('FINAL_MODEL_RUNTIME_HASHES_UNCHANGED',len(model_results),len(runtime['files']),flush=True)


if __name__=='__main__':
    main()
