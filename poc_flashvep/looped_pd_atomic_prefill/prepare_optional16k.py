"""Add a fresh optional calibration source without changing frozen pools."""
import hashlib
import zipfile
from .common import *


def main():
    cfg=config();folder=ROOT/'raw'/cfg['name'];write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        original=json.loads((ROOT/'data/long_source_pool.json').read_text())
        calibration=json.loads((ROOT/'data/calibration_ouro_L16384.json').read_text())
        test=json.loads((ROOT/'data/test_ouro_L16384.json').read_text())
        assert len(calibration)==7 and len(test)==53,'Optional extension requires the recorded frozen pool'
        provenance=json.loads((ROOT/'configs/dataset_provenance.json').read_text())['supplement']
        archive=Path(hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',revision=provenance['revision'],local_files_only=True))
        assert digest(archive)==provenance['source_SHA256']
        tokenizer=AutoTokenizer.from_pretrained(str(MODEL_PATHS['ouro']),trust_remote_code=True,local_files_only=True)
        seen={r['source_SHA256'] for r in original};prefixes={tuple(r['input_ids']['ouro'][:128]) for r in original};chosen=None
        with zipfile.ZipFile(archive) as compressed:
            path=next(n for n in compressed.namelist() if n.endswith('/gov_report.jsonl') or n=='gov_report.jsonl')
            for line in compressed.read(path).decode().splitlines():
                value=json.loads(line);content='Document:\n'+value['context']+'\nQuestion:\n'+value.get('input','')
                messages=[dict(role='user',content=content)]
                sha=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
                if sha in seen:continue
                encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
                ids=list(encoded['input_ids'] if isinstance(encoded,dict) else encoded)
                if len(ids)<16384 or tuple(ids[:128]) in prefixes:continue
                chosen=dict(prompt_id='longbench_gov_report_'+str(value.get('_id',sha[:16])),source_SHA256=sha,
                    source_dataset='LongBench/gov_report',messages=messages,original_token_length=len(ids),input_ids={'ouro':ids[:16384]},
                    source_type='LENGTH_MATCHED_CONTROLLED_PUBLIC_TEXT_PREFIX',length=16384,truncation_explicit=True)
                break
        assert chosen is not None,'No fresh legal 16K calibration source exists in the pinned archive'
        extension='calibration_ouro_L16384_extension_v1.json';source='optional16k_source_extension_v1.json'
        for path,value in [(ROOT/'data'/source,chosen),(ROOT/'data'/extension,calibration+[chosen])]:
            assert not path.exists(),'Never replace frozen or extension data';write(path,value)
        assert not {tuple(r['input_ids']['ouro'][:128]) for r in calibration+[chosen]}.intersection(tuple(r['input_ids']['ouro'][:128]) for r in test)
        write(ROOT/'state/optional16k_data_extension.json',dict(status='CPU_DATA_READY_NOT_GPU_QUALIFIED',
            calibration_file=extension,calibration_count=8,test_file='test_ouro_L16384.json',test_count=53,
            source_file=source,source_revision=provenance['revision'],source_archive_SHA256=provenance['source_SHA256'],
            original_pools_unchanged=True,calibration_test_prefix_overlap=0,
            scope='One fresh public source supplements calibration only. No 16K inference, capacity, or context qualification is established by data preparation.'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise


if __name__=='__main__':main()
