"""Capture installed support and immutable public dataset metadata."""
import dataclasses
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import time
from .common import ROOT,REPO,PRIOR_REPORTS,write,digest,initialize,preserve

DATASETS=('TIGER-Lab/Mantis-Eval','MUIRBENCH/MUIRBENCH','lmms-lab/MP-DocVQA',
    'yubo2333/MMLongBench-Doc','ahmed-masry/ChartQA','HuggingFaceM4/ChartQA','lmms-lab/DocVQA')


def normalize(value):
    if dataclasses.is_dataclass(value):return dataclasses.asdict(value)
    return value


def main():
    initialize()
    import vllm
    from huggingface_hub import HfApi
    runtime=Path(vllm.__file__).parent
    names=('vllm','torch','torchvision','transformers','huggingface_hub','Pillow','nixl','nixl_cu12','pyarrow','pypdfium2','PyMuPDF')
    versions={}
    for name in names:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]='NOT_INSTALLED'
    paths=[runtime/'config/ec_transfer.py',runtime/'config/multimodal.py',
        runtime/'model_executor/models/qwen2_5_vl.py',runtime/'v1/core/sched/scheduler.py',
        runtime/'v1/worker/gpu_model_runner.py',runtime/'v1/engine/input_processor.py']
    paths+=list((runtime/'distributed/ec_transfer').rglob('*.py'))
    paths+=list((runtime/'distributed/kv_transfer/kv_connector/v1/nixl').rglob('*.py'))
    ledger=[]
    for path in paths:
        if not path.exists():continue
        target=ROOT/'references/installed_runtime'/path.relative_to(runtime)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        ledger.append(dict(original=str(path),copy=str(target.relative_to(REPO)),SHA256=digest(path)))
    upstream=Path('/home/esjung/external/vllm-cp-composition-audit')
    source_paths=list((upstream/'examples/disaggregated/disaggregated_encoder').glob('*'))
    source_paths.append(upstream/'docs/features/disagg_encoder.md')
    for path in source_paths:
        if not path.is_file():continue
        target=ROOT/'references/upstream_examples'/path.name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target);ledger.append(dict(original=str(path),copy=str(target.relative_to(REPO)),SHA256=digest(path)))
    write(ROOT/'references/runtime_support_audit.json',dict(time=time.time(),versions=versions,installed_path=str(runtime),
        reference_clone_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip(),files=ledger,
        notes=['Installed v0.26.0 provides ECExampleConnector and ECCPUConnector,not the newer NIXL EC connector described in the November blog.',
            'ECCPUConnector is local engine offload. Its independent engine-local index is not a remote E-to-P registry.',
            'ECExampleConnector is an official debug file handoff. Online production and precomputed embedding replay are different contracts.',
            'Reference clone content is captured with hashes. Its HEAD is not automatically the installed runtime version.']))
    api=HfApi();results=[]
    for name in DATASETS:
        path=ROOT/'references/datasets'/(name.replace('/','__')+'.json');path.parent.mkdir(exist_ok=True)
        if path.exists():results.append(json.loads(path.read_text()));continue
        try:
            info=api.dataset_info(name,files_metadata=True)
            row=dict(name=name,accessible=True,revision=info.sha,card=normalize(info.card_data),
                siblings=[dict(name=s.rfilename,size=s.size,LFS=normalize(s.lfs)) for s in info.siblings])
            if row['card'] is not None and not isinstance(row['card'],(str,dict,list)):row['card']=row['card'].to_dict()
        except Exception as error:row=dict(name=name,accessible=False,error_type=type(error).__name__,error=str(error))
        write(path,row);results.append(row)
        print('DATASET_SOURCE',name,row['accessible'],row.get('revision'),flush=True)
    write(ROOT/'references/dataset_availability_audit.json',dict(time=time.time(),datasets=results,
        scope='Metadata availability only. Media access,processor feasibility and actual natural lengths remain separate gates.'))
    write(ROOT/'references/prior_report_hashes.json',{name:digest(REPO/'poc_flashvep/reports'/name) for name in PRIOR_REPORTS})
    preserve();print('SOURCE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
