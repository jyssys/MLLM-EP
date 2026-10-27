"""Official native loading. IFM uses the evaluator-supported FA3 backend."""
import json
import os
import subprocess
import sys
import time
import torch
from .common import MODEL_ASSETS,XLLM
from ..looped_moe_headroom.adapters import Adapter as PreviousAdapter

class Adapter(PreviousAdapter):
    def __init__(self,name):
        if name=='foil1':
            super().__init__(name)
            return
        assert name in ['ifm_ouro','ifm_huginn']
        assert torch.cuda.device_count()==1
        assert os.environ.get('ENABLE_FLASH_ATTENTION_3')=='true'
        assert os.environ.get('ENABLE_FLASH_ATTENTION_4')=='false'
        self.name=name;self.device=torch.device('cuda:0')
        torch.manual_seed(20261026)
        sys.path.insert(0,str(XLLM))
        from xllm.paper_part1.native_inference import load_native_model
        from xllm.paper_part1.evaluator import _selected_flash_binary_identity
        self.backend_identity=_selected_flash_binary_identity()
        assert self.backend_identity['backend']=='flash_attn_3'
        start=time.perf_counter()
        path=MODEL_ASSETS/name
        self.model,self.tokenizer,self.config=load_native_model(path,device=self.device)
        self.model.eval()
        self.load_seconds=time.perf_counter()-start
        self.config_dict=json.loads((path/'config.json').read_text())
        self.loops=self.model.loop_times
        self.start,self.end=self.model.loop_start_layer,self.model.loop_end_layers
        self.sites={key:module for key,module in self.model.named_modules() if type(module).__name__=='NormalizedMoE'}
        self.recurrent={key for key,module in self.sites.items() if self.start<=module.layer_id<self.end}
        self.backend='Official native xLLM, evaluator-supported FlashAttention3'
        self.controller=None
        assert all(p.dtype==torch.bfloat16 and p.device==self.device for p in self.model.parameters())
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        own=[line.split(',')[0].strip() for line in active.splitlines() if int(line.split(',')[1])==os.getpid()]
        assert len(own)==1
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(line.split(',')[0]):line.split(',')[1].strip() for line in inventory.splitlines()}
        physical=int(os.environ['CUDA_VISIBLE_DEVICES'])
        assert own[0]==mapping[physical],('PHYSICAL_GPU_UUID_MISMATCH',physical,own[0],mapping[physical])
        self.device_uuid=own[0]

    def metadata(self):
        data=super().metadata()
        data['allocator_configuration']=os.environ.get('PYTORCH_CUDA_ALLOC_CONF','DEFAULT')
        if hasattr(self,'backend_identity'):data['backend_binary_identity']=self.backend_identity
        return data
