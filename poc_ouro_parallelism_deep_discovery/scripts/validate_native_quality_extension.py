"""Admit complete native baseline arms without evaluating answers."""
import argparse
import datetime as dt
import gzip
import json
from pathlib import Path
from poc_ouro_parallelism_deep_discovery.contract import ROOT,MODELS,read,sha
from poc_ouro_parallelism_deep_discovery.scripts.native_quality_extension import require


def check_record(row,value,tok,size,protocol_hash,contract_hash):
    require((value['id'],value['dataset'],value['size'],value['mode'],value['TP'])==
        (row['id'],row['dataset'],size,'fixed',4),'Native request identity differs')
    require(value['revision']==MODELS[size]['revision'] and value['physical_gpus']==[4,5,6,7],
        'Native model or physical mask differs')
    require(value['source_protocol_sha256']==protocol_hash and value['extension_contract_sha256']==contract_hash,
        'Native request provenance differs')
    require(value['cap']==8192 and value['temperature']==0 and value['EOS']==tok.eos_token_id==0,
        'Native generation settings differ')
    tokens=value['output_ids']
    require(0<len(tokens)<=8192 and len(tokens)==value['generated_tokens'],'Native output count differs')
    require(all(type(t) is int and 0<=t<49152 for t in tokens),'Invalid native token ID')
    require(value['prompt_ids']==tok.encode(row['prompt'],add_special_tokens=False),'Native prompt differs')
    require(value['text']==tok.decode(tokens,skip_special_tokens=False),'Native text is unbound to raw IDs')
    require(0 not in tokens[:-1],'Native output continues after EOS')
    finish,stop=value['native_finish_reason'],value['native_stop_reason']
    if finish=='length':
        expected='length'
        require(len(tokens)==8192 and tokens[-1]!=0,'Native length cap differs')
    else:
        require(finish=='stop','Unknown native completion')
        expected='eos' if stop in (None,0) else 'stop'
        if expected=='eos':
            require(tokens[-1]==0,'Native EOS metadata lacks the terminal EOS token')
        else:
            require(stop in row.get('stop_strings',[]) and bool(stop),'Unfrozen native stop')
            require(value['native_api_text'].endswith(stop) and stop in value['text'],
                'Native stop metadata lacks the frozen string')
    require(value['finish_reason']==expected,'Derived native completion differs')
    require(value['evidence_class']=='QUALITY_ONLY','Native evidence class differs')
    return dict(prompt_matches=True,text_matches=True,finish_matches=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Use a fresh integrity path')
    freeze_path=ROOT/'artifacts/native_quality_extension_contract_v2.json'
    protocol_path=ROOT/'artifacts/baseline_extension_protocol_v1.json'
    freeze,protocol=read(freeze_path),read(protocol_path)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (freeze_path,protocol_path)}
    for relative,digest in freeze['source_sha256'].items():
        require(sha(ROOT/relative)==digest,'Frozen source changed: '+relative)
        hashes[relative]=digest
    from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
    configure_native_imports()
    import transformers
    from transformers import AutoTokenizer
    require(transformers.__version__=='5.16.1','Use the native generation tokenizer version')
    jobs=[]
    for size in MODELS:
        tag=size.replace('.','').replace('B','')
        run=ROOT/f'outputs/native_quality_extension_base{tag}_TP4_v2_isolated'
        sub=run/('native_extension_'+size)
        required=(run/'guard_status.json',sub/'status.json')
        job=dict(size=size,status='NOT_AVAILABLE',records=[])
        jobs.append(job)
        if not all(p.exists() and read(p).get('success') is True for p in required):
            continue
        require(read(required[0])['exit_code']==0,'Native guard exit differs')
        require(read(required[1])['N']==64,'Native arm count differs')
        launch=read(run/'run_contract.json')
        require(launch['mask']=='4,5,6,7','Native guard mask differs')
        receipt=read(run/'launch_sources.json')
        require(receipt['before_native_imports'] is True and receipt['manifest_sha256']==sha(freeze_path)
            and receipt['observed_sha256']==freeze['source_sha256'],'Native launch-source receipt differs')
        cache=read(run/'cache_environment.json')
        require(cache['set_before_native_imports'] is True,'Native cache setup timing differs')
        require(cache['variables']=={key:str(run/directory) for key,directory in (
            ('VLLM_CACHE_ROOT','vllm_cache'),('TORCHINDUCTOR_CACHE_DIR','inductor_cache'),
            ('TRITON_CACHE_DIR','triton_cache'))},'Native cache ownership differs')
        for relative,digest in freeze['source_sha256'].items():
            if relative.endswith('.py') and not relative.startswith('../'):
                require(launch['source_sha256'].get(relative)==digest,'Launch source differs: '+relative)
        tok=AutoTokenizer.from_pretrained(MODELS[size]['path'],trust_remote_code=True,local_files_only=True)
        frozen={r['id']:r for r in protocol['rows']}
        files=sorted(sub.glob('*.json.gz'))
        require(len(files)==64,'Native raw count differs')
        seen=set()
        for file in files:
            with gzip.open(file,'rt',encoding='utf-8') as stream:
                value=json.load(stream)
            require(value['id'] in frozen and value['id'] not in seen,'Native ID missing or repeated')
            seen.add(value['id'])
            checks=check_record(frozen[value['id']],value,tok,size,sha(protocol_path),sha(freeze_path))
            relative=str(file.relative_to(ROOT))
            hashes[relative]=sha(file)
            job['records'].append(dict(id=value['id'],source=relative,sha256=hashes[relative],**checks))
        require(seen==set(frozen),'Native cohort differs')
        for path in (*required,run/'run_contract.json',run/'launch_sources.json',run/'cache_environment.json',sub/'engine_config.json',sub/'ownership.json'):
            hashes[str(path.relative_to(ROOT))]=sha(path)
        job['status']='PASS'
    result=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),jobs=jobs,input_sha256=hashes,
        transformers=transformers.__version__,evidence_class='PROVENANCE_ONLY',
        rule='Recorded source, prompt, token/text, and completion checks. No independent model re-execution or answer grading.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(complete_arms=sum(j['status']=='PASS' for j in jobs),records=sum(len(j['records']) for j in jobs))))


if __name__=='__main__':
    main()
