"""Read-only package reuse in a separate interpreter and job environment."""
import subprocess
import sys
from poc_ouro_parallelism_deep_discovery.contract import *

def configure_imports():
    # The dedicated interpreter installs or updates no shared dependency.
    # Existing pinned binary packages remain read-only library roots.
    native=Path(NATIVE_PYTHON).parents[1]/'lib/python3.14/site-packages'
    sys.path.insert(0,str(native))
    sys.path.insert(0,str(CDB/'src'))
    sys.path.insert(0,str(SHARED_DEPS/'python_cuda129'))
    sys.path.insert(0,str(SHARED_DEPS/'cdb_dependencies'))

def configure_native_imports():
    """The tested vLLM environment uses its own pinned Transformers, not CDB's version."""
    native=Path(NATIVE_PYTHON).parents[1]/'lib/python3.14/site-packages'
    sys.path.insert(0,str(native))
    # Native FlashInfer can invoke the existing Ninja executable at first use.
    # The isolated interpreter has no installed binaries of its own.
    os.environ['PATH']=str(Path(NATIVE_PYTHON).parent)+':'+os.environ.get('PATH','')

def load_engine(size,mode,*,graphs=False,batch=1,blocks=640,max_model_len=10240,refill=False):
    mask=validate_mask()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CDB,text=True).strip()==CDB_REVISION
    configure_imports()
    import torch
    import transformers
    from safetensors import safe_open
    from looped_cdb.eval.model_loading import load_ouro_tokenizer
    from looped_cdb.models.ouro import OuroConfig,OuroForCausalLM
    from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig,ContinuousDepthBatchingEngine
    assert transformers.__version__=='5.9.0'
    info=MODELS[size]; path=info['path']
    config=OuroConfig.from_pretrained(path,pad_token_id=None)
    config.pad_token_id=None; config.total_ut_steps=4
    assert config.num_hidden_layers==info['layers']
    assert config.hidden_size==2048 and config.intermediate_size==5632
    assert config.num_attention_heads==config.num_key_value_heads==16
    model=OuroForCausalLM.from_pretrained(path,config=config,dtype=torch.bfloat16,
        attn_implementation='paged|flash_attention_3').to('cuda:0').eval()
    model.set_attn_implementation('paged|flash_attention_3')
    configure_kv_cache_policy(model,kv_policy='last_exited')
    for parameter in model.parameters(): parameter.requires_grad_(False)
    q,min_R=(1.,4) if mode=='fixed' else (.5,2)
    cfg=ContinuousDepthBatchingConfig(max_recurrent_steps=4,min_recurrent_steps=min_R,
        exit_threshold=q,synthetic_exit_replay=False,delay_gate_consumption=False,
        use_async_batching=False,use_cuda_graph=graphs,use_cuda_graph_prefill=False,
        max_num_seqs=batch,max_num_batched_tokens=4096,max_model_len=max_model_len,
        num_blocks=blocks,kv_policy='last_exited',kv_pressure_mode='reserve',refill=refill)
    engine=ContinuousDepthBatchingEngine.from_model(model,cfg)
    assert engine.model_adapter._exit_gate_module is model.model.early_exit_gate
    with safe_open(path/'model.safetensors',framework='pt',device='cpu') as checkpoint:
        for field in ['weight','bias']:
            assert torch.equal(getattr(model.model.early_exit_gate,field).detach().cpu(),
                checkpoint.get_tensor('model.early_exit_gate.'+field).to(torch.bfloat16))
    assert len(engine.cache.key_cache)==4*info['layers']
    assert len(engine.cache.value_cache)==4*info['layers']
    uuid=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert uuid==UUIDS[GPUS.index(mask[0])]
    return model,load_ouro_tokenizer(str(path)),engine
