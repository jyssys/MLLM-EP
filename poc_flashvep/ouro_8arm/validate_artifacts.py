"""Read-only validation of protected evidence and completed task records."""
from .contract import *


def main():
    protected=read(ROOT/'source_hashes.json')
    changed=[p for p,digest in protected.items() if not (PROJECT/p).exists() or sha(PROJECT/p)!=digest]
    assert not changed,changed
    assert sha(ROOT/'spec_frozen.md')==read(ROOT/'run_manifest.json')['spec_sha256']
    assert sha(MODEL/'model.safetensors')==read(ROOT/'environment.json')['checkpoint_sha256']
    manifest={r['id']:r for r in dataset(False)};ids={};count=0;invalid=[]
    for folder in (ROOT/'raw').iterdir():
        for file in folder.glob('*.json'):
            try:r=read(file)
            except json.JSONDecodeError:continue
            if not isinstance(r,dict) or r.get('id') not in manifest or 'output_ids' not in r:continue
            assert r['prompt_sha256']==manifest[r['id']]['prompt_sha256'],file
            if r['prompt_ids']!=manifest[r['id']]['input_ids']:
                invalid.append(str(file.relative_to(ROOT)))
                assert folder.name=='quality_S4_pilot_v1',file
                continue
            if r['id'] in ids:assert r['prompt_ids']==ids[r['id']],file
            ids[r['id']]=r['prompt_ids'];assert len(r['output_ids'])==r['output_tokens'],file
            assert r['output_ids'] and all(0<=i<49152 for i in r['output_ids']),file
            assert r['E2E_s']>0,file
            if 'depths' in r:
                assert len(r['depths'])==len(r['output_ids']),file
                assert r['depths'][0]==4 and all(d in [2,3,4] for d in r['depths']),file
            count+=1
    write(ROOT/'state/artifact_validation.json',dict(protected_files=len(protected),changed=[],task_records_checked=count,
        unique_question_inputs=len(ids),valid_records_prompt_ID_parity=True,preserved_input_mismatches=invalid,
        scope='Data integrity only. Preserved input mismatches are not valid cross-arm samples. Not a numerical or quality PASS.'))
    print('ARTIFACTS_PASS',len(protected),count,flush=True)


if __name__=='__main__':main()
