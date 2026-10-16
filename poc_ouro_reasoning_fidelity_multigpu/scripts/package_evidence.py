"""Package immutable raw evidence. Preserve every original local file."""
import tarfile
import csv
import io
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def bundle(paths, target):
    assert not target.exists(), 'Do not overwrite an earlier evidence archive'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, 'x:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    return dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target),
                files=[dict(relative_path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                            sha256=sha(p)) for p in paths])


def main():
    assert (ROOT/'results/quality_v1/guard_status.json').exists(), 'Do not archive an active writer'
    arrays=sorted((ROOT/'results').rglob('*.npz'))
    outputs=sorted(set((ROOT/'results/quality_v1').glob('*_wave*.json')) |
                   set((ROOT/'results/prompt_diagnostic_v1').glob('diagnostic_*_wave*.json')))
    assert arrays and outputs
    manifest=ROOT/'evidence/manifest.json'
    assert not manifest.exists()
    array_bundle=bundle(arrays, ASSETS/'raw_evidence/fidelity_state_arrays_v1.tar.gz')
    text_bundle=bundle(outputs, ROOT/'evidence/natural_generations_v1.tar.gz')
    environment=ROOT/'results/environment.json'
    raw_environment=read(environment)
    processes=list(csv.DictReader(io.StringIO(raw_environment['processes']),skipinitialspace=True))
    assigned={'GPU-'+uuid for uuid in UUIDS}
    public_environment={k:v for k,v in raw_environment.items() if k!='processes'}
    public_environment.update(assigned_processes=[row for row in processes if row['gpu_uuid'] in assigned],
        unassigned_process_count=sum(row['gpu_uuid'] not in assigned for row in processes),
        raw_environment_sha256=sha(environment),
        publication_note='Unassigned process paths and identities remain in the preserved local raw audit, not in Git.')
    public_target=ROOT/'results/environment_public.json'
    assert not public_target.exists()
    write(public_target,public_environment)
    clean_starts=sorted((ROOT/'results').rglob('clean_start.json'))
    for path in clean_starts:
        raw=read(path)
        process_rows=list(csv.DictReader(io.StringIO(raw.get('all_GPU_processes','')),
            fieldnames=['gpu_uuid','pid','process_name'],skipinitialspace=True))
        public={k:v for k,v in raw.items() if k!='all_GPU_processes'}
        public.update(assigned_processes=[row for row in process_rows if row['gpu_uuid'] in assigned],
            unassigned_process_count=sum(row['gpu_uuid'] not in assigned for row in process_rows),
            raw_clean_start_sha256=sha(path),
            publication_note='The complete audit remains local. Unassigned process identities are not published.')
        target=path.with_name('clean_start_public.json')
        assert not target.exists()
        write(target,public)
    environment_bundle=bundle([environment,*clean_starts],ASSETS/'raw_evidence/environment_v1.tar.gz')
    write(manifest,dict(
        original_local_files_preserved=True,
        state_arrays=dict(storage='OFF_GIT_ON_EXPERIMENT_SERVER', **array_bundle),
        natural_outputs=dict(storage='IN_GIT_COMPRESSED_ARCHIVE', **text_bundle),
        environment=dict(storage='OFF_GIT_ON_EXPERIMENT_SERVER',**environment_bundle,
            public_record=str(public_target.relative_to(ROOT))),
        extraction='Extract into a new empty directory. Archive paths start with results/.',
        reconstruction='Use the pinned interpreters, frozen token IDs, and guarded scripts to rerun missing off-Git states.',
        limitation='The off-Git array bundle is not remotely published. Local originals and their hashes remain available.'))
    print('Packaged arrays:',len(arrays),'natural waves:',len(outputs))


if __name__=='__main__':
    main()
