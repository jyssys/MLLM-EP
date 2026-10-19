"""Post-launch provenance checks. This script neither parses nor grades answers."""
import argparse
import datetime as dt
import gzip
import json
from pathlib import Path

from poc_ouro_parallelism_deep_discovery.contract import MODELS, ROOT, read, sha


ARMS = [('1.4B', 'fixed', 4), ('1.4B', 'adaptive', 5), ('2.6B', 'fixed', 6), ('2.6B', 'adaptive', 7)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_record(frozen, value, tokenizer):
    expected_prompt = tokenizer.encode(frozen['prompt'], add_special_tokens=False)
    require(value['prompt_ids'] == expected_prompt, 'Recorded prompt does not encode the frozen prompt')
    tokens = value['output_ids']
    require(value['text'] == tokenizer.decode(tokens, skip_special_tokens=False),
            'Recorded text does not decode the recorded output IDs')
    require(bool(tokens), 'Empty generation')
    stops = [tokenizer.encode(text, add_special_tokens=False) for text in frozen.get('stop_strings', [])]
    require(all(stops), 'Frozen stop encodes an empty sequence')
    matched = [sequence for sequence in stops if len(tokens) >= len(sequence) and tokens[-len(sequence):] == sequence]
    expected_reason = 'eos' if tokens[-1] == 0 else 'length' if len(tokens) == 8192 else 'stop'
    require(value['finish_reason'] == expected_reason, 'Recorded finish precedence differs')
    if expected_reason == 'stop':
        require(bool(matched), 'Stop completion has no frozen token suffix')
    require(0 not in tokens[:-1], 'Generation continues after EOS')
    for sequence in stops:
        length = len(sequence)
        require(not any(tokens[start:start+length] == sequence for start in range(len(tokens)-length)),
                'Generation continues after a frozen stop')
    return dict(prompt_matches=True, text_matches=True, finish_matches=True,
                matched_frozen_stop_count=len(matched), no_earlier_terminal=True)


def validate(root, tokenizer_loader):
    root=Path(root)
    launch_path=root/'artifacts/baseline_extension_launch_contract_v1.json'
    protocol_path=root/'artifacts/baseline_extension_protocol_v1.json'
    launch,protocol=read(launch_path),read(protocol_path)
    require(launch['status']=='FROZEN_BEFORE_GPU_LAUNCH', 'Launch anchor status differs')
    require(launch['requests_per_arm']==protocol['N']==64, 'Launch request count differs')
    require(launch['maximum_new_tokens_per_request']==protocol['quality_cap']==8192, 'Launch cap differs')
    hashes={str(p.relative_to(root)):sha(p) for p in (launch_path,protocol_path)}
    for relative,expected in launch['source_sha256'].items():
        require(sha(root/relative)==expected, 'A source differs from its frozen launch anchor')
        hashes[relative]=expected
    frozen={r['id']:r for r in protocol['rows']}
    require(len(frozen)==64, 'Frozen request IDs repeat')
    jobs=[]
    tokenizer_info={}
    for size,mode,gpu in ARMS:
        short=size.replace('.','').replace('B','')
        directory=root/f'outputs/baseline_extension_base{short}_{mode}_gpu{gpu}_v1'
        sub=directory/'baseline_extension'
        status_path,guard_path=sub/'status.json',directory/'guard_status.json'
        admission=all(p.exists() and read(p).get('success') is True for p in (status_path,guard_path))
        row=dict(size=size,mode=mode,physical_gpu=gpu,source=str(directory.relative_to(root)),
                 status='NOT_AVAILABLE',records=[])
        if not admission:
            row['reason']='Runner and outer guard are not both successful'
            jobs.append(row)
            continue
        require(read(guard_path)['exit_code']==0, 'Successful guard has a nonzero exit')
        status=read(status_path)
        require(status['N_completed']==status['N_expected']==64, 'Runner success count differs')
        require(status['source_protocol_sha256']==hashes[str(protocol_path.relative_to(root))], 'Runner protocol differs')
        require(read(sub/'frozen_protocol.json')==protocol, 'Execution protocol copy differs')
        require(dict(physical_gpu=gpu,size=size,mode=mode) in launch['physical_mapping'], 'Launch mapping differs')
        tokenizer=tokenizer_loader(size)
        require(tokenizer.eos_token_id==0, 'Tokenizer EOS differs')
        info=MODELS[size]
        files={name:sha(info['path']/name) for name in
               ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt')}
        tokenizer_info[size]=dict(revision=info['revision'],path=str(info['path']),
                                  class_name=type(tokenizer).__name__,file_sha256=files)
        paths=sorted(sub.glob('*.json.gz'))
        require(len(paths)==64, 'Complete arm raw count differs')
        seen=set()
        for path in paths:
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                value=json.load(stream)
            require(value['id'] in frozen and value['id'] not in seen, 'Unexpected or repeated record ID')
            require((value['size'],value['mode'],value['physical_gpu'],value['revision'])==
                    (size,mode,gpu,info['revision']), 'Record model/mode/GPU pin differs')
            require(value['source_protocol_sha256']==hashes[str(protocol_path.relative_to(root))], 'Record protocol differs')
            seen.add(value['id'])
            result=check_record(frozen[value['id']],value,tokenizer)
            relative=str(path.relative_to(root))
            hashes[relative]=sha(path)
            row['records'].append(dict(id=value['id'],source=relative,sha256=hashes[relative],**result))
        require(seen==set(frozen), 'Complete arm ID set differs')
        row['status']='PASS'
        jobs.append(row)
    return dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='POST_LAUNCH_INTEGRITY_SUPPLEMENT',jobs=jobs,input_sha256=hashes,tokenizers=tokenizer_info,
        rule='No scoring, sample, cap, gate, EOS, stop, or generation-policy change. No answer correctness is evaluated.',
        limit='This validates recorded provenance, not an independent re-execution of the model or a candidate quality claim.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(), 'Use a new output path')
    from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
    configure_imports()
    import transformers
    from transformers import AutoTokenizer
    require(transformers.__version__=='5.9.0', 'Use the generation tokenizer version')
    loaded={}
    def tokenizer_loader(size):
        if size not in loaded:
            loaded[size]=AutoTokenizer.from_pretrained(str(MODELS[size]['path']),
                trust_remote_code=True,local_files_only=True)
        return loaded[size]
    value=validate(ROOT,tokenizer_loader)
    value['transformers_version']=transformers.__version__
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(complete_arms=sum(j['status']=='PASS' for j in value['jobs']),
                         checked_records=sum(len(j['records']) for j in value['jobs']))))


if __name__=='__main__':
    main()
