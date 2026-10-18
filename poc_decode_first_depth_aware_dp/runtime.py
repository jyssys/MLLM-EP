"""Load the existing official CDB engine. No serving engine is implemented here."""
import os
import subprocess
import sys
import dataclasses
from poc_decode_first_depth_aware_dp.contract import *


def stats_dict(stats):
    # dataclasses.asdict reconstructs Counter from (key, value) pairs and corrupts its keys.
    # Read the actual counter values instead. This changes serialization only.
    def plain(value):
        if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        return value
    return {f.name:plain(getattr(stats,f.name)) for f in dataclasses.fields(stats)}


def configure_imports():
    assert (ASSETS/'cdb_dependencies/transformers/__init__.py').is_file(),'Wait for the isolated dependency installation.'
    sys.path.insert(0,str(CDB/'src'))
    sys.path.insert(0,str(ASSETS/'python_cuda129'))
    sys.path.insert(0,str(ASSETS/'cdb_dependencies'))
    import transformers
    assert transformers.__version__=='5.9.0',transformers.__version__


def load_engine(q=.5,*,graphs=False,batch=1,blocks=640,refill=True,max_model_len=8192):
    assert set(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))<={4,5,6,7}
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CDB,text=True).strip()==CDB_REVISION
    configure_imports()
    import torch
    from looped_cdb.eval.model_loading import load_ouro_tokenizer
    from looped_cdb.models.ouro import OuroConfig,OuroForCausalLM
    from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig,ContinuousDepthBatchingEngine
    # Transformers 4.57 supplied this default implicitly. Transformers 5.9 does not.
    # The pinned ByteDance config omits it. No padding row or model weight changes.
    config=OuroConfig.from_pretrained(MODEL,pad_token_id=None)
    config.pad_token_id=None
    config.total_ut_steps=4
    model=OuroForCausalLM.from_pretrained(MODEL,config=config,dtype=torch.bfloat16,
        attn_implementation='paged|flash_attention_3').to('cuda').eval()
    model.set_attn_implementation('paged|flash_attention_3')
    configure_kv_cache_policy(model,kv_policy='last_exited')
    tokenizer=load_ouro_tokenizer(str(MODEL))
    cfg=ContinuousDepthBatchingConfig(max_recurrent_steps=4,min_recurrent_steps=2,
        exit_threshold=q,synthetic_exit_replay=False,delay_gate_consumption=False,
        use_async_batching=False,use_cuda_graph=graphs,use_cuda_graph_prefill=False,
        max_num_seqs=batch,max_num_batched_tokens=4096,max_model_len=max_model_len,
        num_blocks=blocks,kv_policy='last_exited',kv_pressure_mode='reserve',refill=refill)
    engine=ContinuousDepthBatchingEngine.from_model(model,cfg)
    assert engine.model_adapter._exit_gate_module is model.model.early_exit_gate
    from safetensors import safe_open
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as checkpoint:
        for field in ['weight','bias']:
            assert torch.equal(getattr(model.model.early_exit_gate,field).detach().cpu(),
                checkpoint.get_tensor('model.early_exit_gate.'+field).to(torch.bfloat16))
    assert len(engine.cache.key_cache)==96 and len(engine.cache.value_cache)==96
    physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))[0]
    uuid=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert uuid==UUIDS[GPUS.index(physical)]
    return model,tokenizer,engine
