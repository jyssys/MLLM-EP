"""Keep real text and a complete generation suffix at each input length."""
import zipfile
from .common import *


def main():
    if (ROOT/'data/variable_qualification.json').exists():print('VARIABLE_DATA_FROZEN');return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    source=json.loads((ROOT/'data/unseen_calibration.json').read_text());wanted={r['prompt_id'] for r in source}
    suffix=json.loads((ROOT/'data/unseen_manifest.json').read_text())['generation_suffix_ids']
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip')
    full={}
    with zipfile.ZipFile(archive) as stream:
        for task in ('gov_report','qmsum','narrativeqa','qasper','multifieldqa_en','hotpotqa','2wikimqa','musique'):
            paths=[n for n in stream.namelist() if n.endswith('/'+task+'.jsonl') or n==task+'.jsonl']
            if not paths:continue
            for index,line in enumerate(stream.read(paths[0]).decode().splitlines()):
                row=json.loads(line);identifier=f'longbench_{task}_{row.get("_id",index)}'
                if identifier not in wanted:continue
                messages=[dict(role='user',content='Document:\n'+row['context']+'\nQuestion:\n'+row.get('input',''))]
                ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                    enable_thinking=False,return_dict=False)
                full[identifier]=(ids,task)
    assert set(full)==wanted
    pools={}
    for length in (2048,4096,8192,12288):
        values=[]
        for identifier,(ids,task) in full.items():
            if len(ids)<length:continue
            clipped=ids[:length-len(suffix)]+suffix
            values.append(dict(prompt_id=identifier+f'__T{length}',source_document_id=identifier,
                source_dataset='LongBench/'+task,input_ids={'qwen8':clipped},length=length,
                original_tokenized_length=len(ids),thinking_disabled_suffix_verified=True,
                truncation_explicit=True,source_type='CONTROLLED_REAL_TEXT_LENGTH_VARIANT'))
        assert len(values)>=8,(length,len(values));pools[length]=values
        write(ROOT/'data'/f'variable_calibration_L{length}.json',values)
    rows=[pools[length][i//4] for i in range(32) for length in [(2048,4096,8192,12288)[i%4]]]
    write(ROOT/'data/variable_qualification.json',rows)
    write(ROOT/'data/variable_manifest.json',dict(model_revision=MODEL_REVISION,archive_SHA256=digest(archive),
        document_pool='unseen_calibration.json only. Confirmation document IDs are excluded.',
        lengths={str(k):dict(count=len(v),SHA256=digest(ROOT/'data'/f'variable_calibration_L{k}.json')) for k,v in pools.items()},
        qualification_SHA256=digest(ROOT/'data/variable_qualification.json'),
        scope='Input-length variants can share source documents. They are not independent prompt samples. No filler or repeated token extension.'))
    print('VARIABLE_REAL_DATA_FROZEN',{k:len(v) for k,v in pools.items()})


if __name__=='__main__':main()
