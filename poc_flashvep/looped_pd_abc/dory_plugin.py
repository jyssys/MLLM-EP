"""Load the unchanged official Dory PR modules in the isolated vLLM environment."""
import importlib.util
import sys
from copy import deepcopy
from .common import ROOT

def load(name,path):
 if name in sys.modules:return sys.modules[name]
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 spec.loader.exec_module(module)
 return module

def install():
 from transformers import AutoConfig
 from vllm.transformers_utils.config import _CONFIG_REGISTRY
 config=load('vllm.transformers_utils.configs.dory',ROOT/'sources/dory_config_876edc.py')
 if not getattr(config.DoryConfig,'_abc_rope_compat',False):
  def attention_only_rope(self,*args,**kwargs):
   # Dory's MLP-only sublayers have no RoPE configuration.
   # The checkpoint's two attention maps already contain all required values.
   if self.rope_parameters is not None:
    assert set(self.rope_parameters)>={'sliding_attention','full_attention'}
  config.DoryConfig.standardize_rope_params=attention_only_rope
  original=config.DoryConfig.__post_init__
  def preserve_nested_rope(self,**kwargs):
   # Transformers 5.16 adds flat default keys to the nested checkpoint map.
   # Preserve the original per-attention map. Do not change either theta.
   supplied=deepcopy(self.rope_parameters)
   original(self,**kwargs)
   if supplied is not None:self.rope_parameters=supplied
   self.has_no_defaults_at_init=True
  config.DoryConfig.__post_init__=preserve_nested_rope
  config.DoryConfig._abc_rope_compat=True
 AutoConfig.register('dory',config.DoryConfig,exist_ok=True)
 _CONFIG_REGISTRY['dory']=config.DoryConfig
 from vllm import ModelRegistry
 ModelRegistry.register_model('DoryForCausalLM','poc_flashvep.looped_pd_abc.dory_plugin:DoryForCausalLM')
 global DoryForCausalLM
 DoryForCausalLM=load('vllm.model_executor.models.dory',ROOT/'sources/dory_vllm_876edc.py').DoryForCausalLM
