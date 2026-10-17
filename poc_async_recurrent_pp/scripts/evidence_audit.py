"""Check final evidence provenance without modifying source artifacts."""
import argparse
import hashlib
import zipfile
from poc_async_recurrent_pp.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wave',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    manifest=read(ROOT/'outputs/manifests/data.json')
    expected={r['id'] for dataset in ['gsm8k','math500']
              for r in [x for x in manifest['pilot'] if x['dataset']==dataset][:16]}
    cases=[]
    ids_by_arm={}
    eos_token=read(MODEL/'config.json')['eos_token_id']
    for gpu,arm in zip(GPUS,['exact','stale','delta','w0']):
        directory=args.wave/arm
        status=read(directory/'status.json')
        config=read(directory/'eval_contract.json')
        assert status['success'] and status['n']==len(expected)
        assert config['GPU_mask']==str(gpu)
        assert config['R']==4 and config['dtype']=='BF16'
        assert config['tokenizer_revision']==REVISION
        assert config['predictor_sha256']==sha(ROOT/'outputs/manifests/predictor.json')
        assert config['data_sha256']==sha(ROOT/'outputs/manifests/data.json')
        assert config['exact_prefill'] and config['sampling']=='Greedy'
        actual=[]
        for identifier in sorted(expected):
            row=read(directory/(identifier.replace('/','_')+'.json'))
            assert row['id']==identifier and row['native_fixed_batch']
            assert row['generated_tokens']==len(row['ids'])<=config['cap']
            assert row['cache_slots']==96 and row['logical_token_writes_after_EOS']==0
            assert row['approximate_KV_persisted']==(arm!='exact')
            assert row['exact_prefill'] and row['cap']==config['cap']
            assert row['finish_reason'] in ['length','eos']
            assert row['truncated']==(row['finish_reason']=='length')
            assert (eos_token in row['ids'])==(row['finish_reason']=='eos')
            if eos_token in row['ids']:
                assert row['ids'].index(eos_token)==len(row['ids'])-1
            actual.append(identifier)
        ids_by_arm[arm]=actual
        cases.append(dict(arm=arm,physical_gpu=gpu,n=len(actual),logical_cuda_rank=0,
            native_generation=True,depth_slots=96,exact_prefill=True,
            candidate_KV_persisted=arm!='exact',logical_generation_after_EOS=0,
            predictor_sha256=config['predictor_sha256']))
    source_checks=[]
    for wave in [ROOT/'outputs/pilot_cap2048_v2',args.wave]:
        with zipfile.ZipFile(wave/'source_snapshot.zip') as archive:
            for name in ['emulator.py','scripts/baseline.py','scripts/eval_quality.py']:
                digest=hashlib.sha256(archive.read(name)).hexdigest()
                assert digest==sha(ROOT/name), 'Primary inference source changed'
                source_checks.append(dict(wave=wave.name,path=name,sha256=digest))
    excluded=['diagnostics_v1','pilot_cap2048_v1','batch_reference_v1']
    assert all((ROOT/'outputs'/name).exists() for name in excluded), 'Preserve rejected evidence'
    write(args.output,dict(pass_gate=True,independent_questions=len(expected),rows=cases,
        source_checks=source_checks,excluded_evidence_preserved=excluded,
        full_GSM8K_1319='NOT_RUN',full_MATH500_500='NOT_RUN',
        physical_distributed_architecture='NOT_RUN. Independent singleton-GPU functional emulators.',
        caveat='Native fixed-batch generation can write unused inactive-padding KV after EOS. These are not logical output tokens.'))
    print('EVIDENCE_AUDIT_PASS',len(expected),'questions',flush=True)


if __name__=='__main__':
    main()
