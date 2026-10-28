# SPDX-License-Identifier: Apache-2.0
# Architecture follows the official Dory vLLM PR at 876edc971507f602b18d1ec523c174a068cc02d7.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Minimal native RLT Dory adapter. This adapter requires independent qualification.

It preserves BF16 weights, nGPT operations, reinjection, sliding windows, and R2.
The runtime stores 36 attention layers, not 72 alternating physical sublayers.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from vllm_rlt.layers import RotaryEmbedding,apply_rotary_pos_emb

def justnorm(x):
 xf=x.float()
 return (xf*torch.rsqrt(xf.square().sum(-1,keepdim=True)+1e-6)).to(x.dtype)

@dataclass(frozen=True)
class DoryRuntimeConfig:
 values:dict
 @classmethod
 def from_dict(cls,v):
  if v['model_type']!='dory' or v['n_recurrent_loops']!=2:raise ValueError('Only pinned Dory R2 is qualified for this adapter')
  return cls(dict(v))
 def to_dict(self):return self.values.copy()
 @property
 def hidden_size(self):return 2*self.values['hidden_size']
 @property
 def num_hidden_layers(self):return sum(x!='mlp' for x in self.values['layer_types'])
 @property
 def num_key_value_heads(self):return self.values['num_key_value_heads']
 @property
 def num_attention_heads(self):return self.values['num_attention_heads']
 @property
 def head_dim(self):return self.values['head_dim']
 @property
 def total_ut_steps(self):return self.values['n_recurrent_loops']
 @property
 def vocab_size(self):return self.values['vocab_size']
 @property
 def max_position_embeddings(self):return self.values['max_position_embeddings']
 @property
 def eos_token_id(self):return (2,11)
 @property
 def bos_token_id(self):return 1

class Attention(nn.Module):
 def __init__(self,c,i,cache_layer):
  super().__init__();self.c=c;self.index=i;self.cache_layer=cache_layer
  H=c['hidden_size'];Q=c['num_attention_heads']*c['head_dim'];K=c['num_key_value_heads']*c['head_dim']
  for name,size in [('q_proj',Q),('k_proj',K),('v_proj',K),('g_proj',Q)]:setattr(self,name,nn.Linear(H,size,bias=False))
  self.o_proj=nn.Linear(Q,H,bias=False)
  self.sqk=nn.Parameter(torch.empty(Q));self.s_gate=nn.Parameter(torch.empty(Q))
  self.rope=RotaryEmbedding(c['head_dim'],c['rope_parameters'][c['layer_types'][i]]['rope_theta'])
 def forward(self,h,batch,cache):
  c=self.c;n=len(h);D=c['head_dim'];heads=c['num_attention_heads'];kv=c['num_key_value_heads']
  if not hasattr(self,'abc_packed_qkv'):
   self.register_buffer('abc_packed_qkv',torch.cat((self.q_proj.weight,self.k_proj.weight,self.v_proj.weight),0),persistent=False)
  packed=F.linear(h,self.abc_packed_qkv)
  q,k,v=packed.split((heads*D,kv*D,kv*D),-1)
  # Reuse the reference's rotary CUDA operator. Separate BF16 products
  # introduce an extra rounding before the rotation sum.
  from vllm import _custom_ops as ops
  cos,sin=self.rope(h,batch.position_ids)
  table=torch.cat((cos[:,0,:D//2],sin[:,0,:D//2]),-1).contiguous()
  ops.rotary_embedding(torch.arange(n,device=h.device,dtype=torch.long),q,k,D,table,True)
  q=q.view(n,heads,D);k=k.view(n,kv,D);v=v.view(n,kv,D)
  scale=(self.sqk.float()*(math.sqrt(D)/c['initializer_range'])).view(1,heads,D)
  q=(justnorm(q).float()*scale.square()).to(h.dtype);k=justnorm(k)
  cache._write_prepared(self.cache_layer,batch,k,v)
  if c['layer_types'][self.index]=='full_attention':out=cache._attend_prepared(self.cache_layer,batch,q)
  else:
   # The native FA4 kernel supplies exact sliding attention. No KV approximation.
   storage=getattr(cache,'cache',cache)
   kw=dict(seqused_k=batch.context_lengths,page_table=batch.block_tables,causal=True,
    max_seqlen_q=batch.max_seqlen_q,max_seqlen_k=batch.block_tables.shape[1]*storage.block_size,
    num_splits=1,window_size=(c['sliding_window']-1,0))
   if batch.cu_seqlens_q is not None:
    kw['cu_seqlens_q']=batch.cu_seqlens_q;query=q
   else:query=q.unsqueeze(1)
   out=storage.attention.kernel.kernel(query,storage.key_cache[:,self.cache_layer],storage.value_cache[:,self.cache_layer],**kw)
   if isinstance(out,tuple):out=out[0]
   if batch.cu_seqlens_q is None:out=out.squeeze(1)
  out=out.reshape(n,-1)
  gate=self.g_proj(h);g0=c.get('attention_output_gate_logit_scale_init') or math.sqrt(c['hidden_size'])
  sg=(self.s_gate.float()*math.sqrt(g0)/c['initializer_range']).square()
  return self.o_proj(justnorm((out.float()*torch.sigmoid(gate.float()*sg)).to(out.dtype)))

class MLP(nn.Module):
 def __init__(self,c):
  super().__init__();self.factor=math.sqrt(c['hidden_size']);I=c['intermediate_size'];H=c['hidden_size']
  self.gate_proj=nn.Linear(H,I,bias=False);self.up_proj=nn.Linear(H,I,bias=False);self.down_proj=nn.Linear(I,H,bias=False)
  self.suv_gate=nn.Parameter(torch.empty(I));self.suv_up=nn.Parameter(torch.empty(I))
 def forward(self,h):
  if not hasattr(self,'abc_packed_gate_up'):
   self.register_buffer('abc_packed_gate_up',torch.cat((self.gate_proj.weight,self.up_proj.weight),0),persistent=False)
  g,u=F.linear(h,self.abc_packed_gate_up).chunk(2,-1)
  gate=(g.float()*(self.suv_gate.float()*self.factor)).to(h.dtype)
  up=(u.float()*(self.suv_up.float()*self.factor)).to(h.dtype)
  return self.down_proj(justnorm((F.silu(gate.float())*up.float()).to(h.dtype)))

class Layer(nn.Module):
 def __init__(self,c,i,cache_layer):
  super().__init__();self.attention=c['layer_types'][i]!='mlp';self.scale=c['ngpt_alpha_init']/c['initializer_range']
  self.mixer=Attention(c,i,cache_layer) if self.attention else MLP(c)
  self.register_parameter('attn_alpha' if self.attention else 'mlp_alpha',nn.Parameter(torch.empty(c['hidden_size'])))
 def forward(self,h,batch,cache):
  out=self.mixer(h,batch,cache) if self.attention else self.mixer(h)
  alpha=self.attn_alpha if self.attention else self.mlp_alpha
  a=justnorm(h).float();b=justnorm(out).float()
  return justnorm(a+(alpha.float()*self.scale).abs()*(b-a)).to(h.dtype)

class DoryForCausalLM(nn.Module):
 requires_boundary_kv=True
 def __init__(self,config):
  super().__init__();self.config=config;c=config.values
  self.backbone=nn.Module();self.backbone.embeddings=nn.Embedding(c['vocab_size'],c['hidden_size'])
  self.backbone.layers=nn.ModuleList();index=0;self.recurrent_kv_layers=[];self.coda_kv_layers=[]
  for i in range(c['num_hidden_layers']):
   self.backbone.layers.append(Layer(c,i,index))
   if c['layer_types'][i]!='mlp':
    if c['n_input_layers']<=i<c['n_input_layers']+c['n_recurrent_layers']:self.recurrent_kv_layers.append(index)
    elif i>=c['n_input_layers']+c['n_recurrent_layers']:self.coda_kv_layers.append(index)
    index+=1
  self.recurrent_kv_layers=tuple(self.recurrent_kv_layers);self.coda_kv_layers=tuple(self.coda_kv_layers)
  self.lm_head=nn.Linear(c['hidden_size'],c['vocab_size'],bias=False)
  self.lm_head.logit_scale=nn.Parameter(torch.empty(c['vocab_size']))
 def prelude_prepared(self,tokens,batch,cache,*,generators=None):
  h=self.backbone.embeddings(tokens)
  for layer in self.backbone.layers[:self.config.values['n_input_layers']]:h=layer(h,batch,cache)
  return torch.cat((h,h),-1)
 def recurrent_prepared(self,hidden,batch,cache,*,compute_gate=True):
  state,inject=hidden.chunk(2,-1);h=justnorm(state+inject);c=self.config.values
  for layer in self.backbone.layers[c['n_input_layers']:c['n_input_layers']+c['n_recurrent_layers']]:h=layer(h,batch,cache)
  gate=torch.full((len(h),),-1e4,device=h.device,dtype=h.dtype) if compute_gate else None
  return torch.cat((justnorm(h),inject),-1),gate
 def coda_prepared(self,hidden,batch,cache,*,compute_logits=True):
  h,_=hidden.chunk(2,-1);c=self.config.values
  for layer in self.backbone.layers[c['n_input_layers']+c['n_recurrent_layers']:]:h=layer(h,batch,cache)
  if not compute_logits:return None
  return self.lm_head(h).float()*(self.lm_head.logit_scale.float()/c['initializer_range'])
 @classmethod
 def from_pretrained(cls,path_or_repo,*,device='cpu',dtype=torch.bfloat16,config=None,**kwargs):
  from safetensors import safe_open
  folder=Path(path_or_repo);config=config or DoryRuntimeConfig.from_dict(json.loads((folder/'config.json').read_text()))
  with torch.device('meta'):model=cls(config)
  expected=dict(model.named_parameters())
  with safe_open(folder/'model.safetensors',framework='pt',device='cpu') as f:
   if set(f.keys())!=set(expected):raise ValueError(f'Dory checkpoint keys differ: {set(f.keys())^set(expected)}')
   for name,param in expected.items():
    tensor=f.get_tensor(name)
    if tuple(tensor.shape)!=tuple(param.shape):raise ValueError('Dory checkpoint shape mismatch: '+name)
    module,key=name.rsplit('.',1);setattr(model.get_submodule(module),key,nn.Parameter(tensor.to(device=device,dtype=dtype),requires_grad=False))
  for module in model.modules():
   if isinstance(module,RotaryEmbedding):module.inv_freq=module.frequencies(device=device)
  return model.eval().requires_grad_(False)

def install():
 import vllm_rlt.models as models
 import vllm_rlt.models.config as configs
 models.MODEL_MAPPING['dory']=DoryForCausalLM
 configs._CONFIG_TYPES['dory']=DoryRuntimeConfig
