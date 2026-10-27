"""Official task scorers. Transport identity is separate from native identity."""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
from .common import *

def evaluate_ifm(a,ep,target,config):
    from xllm.distributed import get_model_parallel_world_size,get_model_parallel_rank
    assert get_model_parallel_world_size()==1 and get_model_parallel_rank()==0, 'Expert parallelism must not silently enable vocabulary TP'
    from xllm.paper_part1.evaluator import evaluate_native_task,native_evaluator_hash,NativeGenerationConfig
    from xllm.paper_part1.eval_protocol import PAPER_PROTOCOL
    from xllm.paper_part1.files import sha256_json
    official_hash=native_evaluator_hash()
    identity=sha256_json(dict(official_evaluator=official_hash,ep_degree=ep.world,
        transport=digest(REPO/'poc_flashvep/looped_moe_ep/runtime.py'),study_sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    suite=sha256_json(dict(tasks=config['tasks'],protocol=PAPER_PROTOCOL.to_dict(),transport=identity))
    output=target/f'official.rank{ep.rank}'
    result=[]
    for task in config['tasks']:
        value=evaluate_native_task(model=a.model,tokenizer=a.tokenizer,task_name=task,tasks_root_dir=DATA/'paper_part1',
            output_dir=output,model_hash=digest(MODEL_ASSETS/config['model']/'artifact_manifest.json'),
            suite_hash=suite,evaluator_hash=identity,generation=NativeGenerationConfig(),task_spec=PAPER_PROTOCOL.spec_for_task(task))
        result.append(asdict(value))
        write(target/f'quality_progress.rank{ep.rank}.json',result)
    write(target/f'official_summary.rank{ep.rank}.json',dict(tasks=result,protocol='Official paper-part1 task specifications',
        official_evaluator_hash=official_hash,transport_identity=identity,ep_degree=ep.world,model_parallel_degree=1))

def evaluate_foil(a,ep,target,config):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm=HFLM(pretrained=a.model,tokenizer=a.tokenizer,device='cuda',batch_size=1,dtype='bfloat16',max_length=4096)
    value=lm_eval.simple_evaluate(model=lm,tasks=config['tasks'],num_fewshot=0,limit=config.get('limit'),
        log_samples=True,bootstrap_iters=1000)
    value['_study_scope']=dict(checkpoint='Foil-1',ep=ep.world if ep else 1,
        model_code='Official checkpoint bundled modeling_loop_lm.py',precision='BF16',
        torch=__import__('torch').__version__,transformers=importlib.metadata.version('transformers'),
        lm_eval=importlib.metadata.version('lm_eval'),datasets=importlib.metadata.version('datasets'),
        native_protocol_comparison='Original HFLM task definitions, 0-shot, original tokenizer, full 4096 context.')
    # The harness can include numpy values and dataclasses in full provenance.
    import numpy as np
    def default(obj):
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,np.ndarray):return obj.tolist()
        if hasattr(obj,'to_dict'):return obj.to_dict()
        return str(obj)
    (target/f'foil_results.rank{ep.rank if ep else 0}.json').write_text(json.dumps(value,default=default,ensure_ascii=False,indent=2)+'\n')

def official_foil(config,target):
    sys.path.insert(0,'/home/esjung/external/how-to-loop-moe-poc')
    from src.eval.run_eval import run_lm_eval_harness
    value=run_lm_eval_harness(MODEL_ASSETS/'foil1/Foil-1',tasks=tuple(config['tasks']),num_fewshot=0,
        limit=config.get('limit'),device='cuda',batch_size=1,tokenizer=str(MODEL_ASSETS/'foil1/Foil-1'),max_length=4096,
        log_samples=True)
    import numpy as np
    def default(obj):
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,np.ndarray):return obj.tolist()
        return str(obj)
    (target/'foil_results.rank0.json').write_text(json.dumps(value,default=default,ensure_ascii=False,indent=2)+'\n')
