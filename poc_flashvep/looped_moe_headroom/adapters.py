"""Native loading and full-prefix forward paths. No substitute architectures."""
import importlib.metadata
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from .common import ASSETS, ROOT, XLLM

class Adapter:
    def __init__(self, name):
        assert torch.cuda.device_count() == 1, 'One visible GPU is mandatory'
        self.name = name
        self.device = torch.device('cuda:0')
        properties=torch.cuda.get_device_properties(0)
        self.device_uuid=str(getattr(properties,'uuid',''))
        inventory=json.loads((ROOT/'hardware_inventory.json').read_text())
        lines=inventory['gpu_query']['stdout'].splitlines()[1:]
        expected={int(line.split(',')[0]):line.split(',')[1].strip() for line in lines}
        physical=int(os.environ['CUDA_VISIBLE_DEVICES'])
        torch.manual_seed(20261026)
        started = time.perf_counter()
        path = ASSETS / name
        if name.startswith('ifm_'):
            os.environ['ENABLE_FLASH_ATTENTION_4'] = 'true'
            os.environ['ENABLE_FLASH_ATTENTION_3'] = 'false'
            sys.path.insert(0, str(XLLM))
            from xllm.paper_part1.native_inference import load_native_model
            self.model, self.tokenizer, self.config = load_native_model(path, device=self.device)
            self.config_dict = json.loads((path / 'config.json').read_text())
            self.loops = self.model.loop_times
            self.start, self.end = self.model.loop_start_layer, self.model.loop_end_layers
            self.sites = {key: module for key, module in self.model.named_modules()
                if type(module).__name__ == 'NormalizedMoE'}
            self.recurrent = {key for key in self.sites
                if self.start <= self.sites[key].layer_id < self.end}
            self.backend = 'native xLLM FlashAttention4 official ENABLE_FLASH_ATTENTION_4 path'
        else:
            assert name == 'foil1'
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model, self.loading_info = AutoModelForCausalLM.from_pretrained(str(path), subfolder='Foil-1',
                trust_remote_code=True, dtype=torch.bfloat16, local_files_only=True,output_loading_info=True)
            for field in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:
                assert not self.loading_info.get(field),self.loading_info
            self.loading_info={k:sorted(v) if isinstance(v,set) else v for k,v in self.loading_info.items()}
            self.model=self.model.to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(str(path / 'Foil-1'), local_files_only=True, trust_remote_code=True)
            self.config = self.model.config
            self.config_dict = self.config.to_dict()
            self.loops = self.config.num_stacks
            self.sites = {key: module for key, module in self.model.named_modules()
                if type(module).__name__ == 'GroupedMoEPrenormBlock'}
            self.recurrent = {key for key in self.sites if '.stack.layers.' in key}
            self.backend = 'official Foil remote code, torch grouped_mm and SDPA'
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        assert self.sites and self.recurrent
        assert all(p.dtype == torch.bfloat16 for p in self.model.parameters())
        assert all(p.device == self.device for p in self.model.parameters())
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        own=[line.split(',')[0].strip() for line in active.splitlines() if int(line.split(',')[1])==os.getpid()]
        assert own==[expected[physical]],('PHYSICAL_GPU_MAPPING_MISMATCH',own,physical,expected[physical])
        self.device_uuid=own[0]
        self.controller = None

    def encode(self, text, length=None):
        if self.name.startswith('ifm_'):
            ids = self.tokenizer.encode(text, bos=True, eos=False)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        if length is not None:
            ids = ids[:length]
        assert len(ids) >= 2
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def forward(self, ids):
        if self.controller is not None:
            self.controller.reset(ids.shape[-1])
        if self.name.startswith('ifm_'):
            logits, _, cache = self.model(ids, multi_segments=False, cache=None)
            assert cache is None
        else:
            out = self.model(input_ids=ids, use_cache=False)
            logits = out.logits
        assert logits.shape[:2] == ids.shape
        if self.controller is not None and self.controller.mode != 'disabled':
            for site in self.sites:
                expected=self.loops if site in self.recurrent else 1
                assert self.controller.calls[site]==expected,(site,self.controller.calls[site],expected)
        return logits

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def eos(self):
        value = getattr(self.tokenizer, 'eos_id', None)
        if value is None:
            value = getattr(self.tokenizer, 'eos_token_id', None)
        return value

    @torch.inference_mode()
    def generate(self, ids, cap=64, stop_eos=True):
        # Both released models use full-prefix recomputation. This is not cached decode.
        output, steps = [], []
        reason = 'length'
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(cap):
            token = self.forward(ids)[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - start)
            value = token.item()
            output.append(value)
            ids = torch.cat([ids, token], dim=-1)
            if stop_eos and value == self.eos():
                reason = 'eos'
                break
        return dict(ids=output, text=self.decode(output), finish_reason=reason,
            truncated=reason == 'length', timestamps_s=steps, e2e_s=steps[-1],
            ttft_s=steps[0], tpot_s=(steps[-1]-steps[0])/(len(steps)-1) if len(steps)>1 else None,
            execution='FULL_PREFIX_RECOMPUTATION', cache=None)

    def metadata(self):
        sites = []
        for key, m in self.sites.items():
            sites.append(dict(site=key, recurrent=key in self.recurrent,
                experts=getattr(m, 'n_experts', getattr(m, 'num_experts', None)),
                topk=getattr(m, 'topk', getattr(m, 'num_active', None)),
                parameter_pointers={k: v.data_ptr() for k,v in m.named_parameters()}))
        return dict(model=self.name, config=self.config_dict, loops=self.loops,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            persistent_state_elements=sum(t.numel() for t in self.model.state_dict().values()),
            registered_buffer_elements=sum(b.numel() for b in self.model.buffers()),
            registered_buffer_bytes=sum(b.numel()*b.element_size() for b in self.model.buffers()),
            buffer_dtypes=sorted({str(b.dtype) for b in self.model.buffers()}),
            loading_info=getattr(self,'loading_info',{'verification':'Official native manifest and strict state load'}),
            parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()),
            physical_moe_sites=sites, recurrent_sites=len(self.recurrent), backend=self.backend,
            generation_execution='FULL_PREFIX_RECOMPUTATION', load_seconds=self.load_seconds,
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), torch=torch.__version__,
            cuda=torch.version.cuda, python=sys.version,
            package_versions={p: importlib.metadata.version(p) for p in ['transformers', 'einops', 'safetensors']},
            visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], device_name=torch.cuda.get_device_name(),
            device_uuid=self.device_uuid, compute_capability=list(torch.cuda.get_device_capability()))

def quality(logits, reference, ids):
    # The Foil labels argument does not shift targets. Never use model.loss here.
    logits, reference = logits.float(), reference.float()
    logp, logq = logits.log_softmax(-1), reference.log_softmax(-1)
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()
    ref_nll = F.cross_entropy(reference[:, :-1].reshape(-1, reference.shape[-1]), ids[:, 1:].reshape(-1)).item()
    agreement=logits.argmax(-1)==reference.argmax(-1)
    equal_count=int(agreement.sum().item())
    return dict(nll=nll, baseline_nll=ref_nll, delta_nll=nll-ref_nll,
        kl_reference_to_variant=(logq.exp() * (logq-logp)).sum(-1).mean().item(),
        top1_agreement=equal_count/agreement.numel(),top1_equal_count=equal_count,
        logit_position_denominator=agreement.numel(),
        token_denominator=ids.numel()-1, finite=bool(torch.isfinite(logits).all()))
