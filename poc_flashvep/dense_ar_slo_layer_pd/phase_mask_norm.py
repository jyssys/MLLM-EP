"""Keep original phase reductions inside a graph-replay-safe normalization."""
import os
from .common import *
from .exact_phase_norm import CUDA as ORIGINAL

CPP=r'''
#include <torch/extension.h>
void masked_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
                 torch::Tensor phases, double epsilon);
'''
HEADER='const int num_tokens, const int hidden_size) {'
assert ORIGINAL.count(HEADER)==1
KERNEL=ORIGINAL.split('\nvoid phase_norm(')[0].replace(HEADER,
    'const int num_tokens, const int hidden_size, const int32_t* phases, int expected) {\n'
    '  if (phases[blockIdx.x] != expected) return;')
CUDA=KERNEL+r'''
void masked_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
                 torch::Tensor phases, double epsilon) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda() && phases.is_cuda());
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && residual.scalar_type()==at::kBFloat16 && weight.scalar_type()==at::kBFloat16);
  TORCH_CHECK(x.dim()==2 && x.size(1)==4096 && x.size(0)>0);
  TORCH_CHECK(x.sizes()==residual.sizes() && residual.is_contiguous() && weight.is_contiguous() && weight.numel()==4096);
  TORCH_CHECK(phases.scalar_type()==at::kInt && phases.is_contiguous() && phases.numel()>=x.size(0));
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream=c10::cuda::getCurrentCUDAStream();
  original_scalar_phase_norm<at::BFloat16><<<x.size(0),256,0,stream>>>(
    x.data_ptr<at::BFloat16>(),x.stride(0),residual.data_ptr<at::BFloat16>(),weight.data_ptr<at::BFloat16>(),
    static_cast<float>(epsilon),x.size(0),4096,phases.data_ptr<int32_t>(),0);
  original_scalar_phase_norm<at::BFloat16><<<x.size(0),1024,0,stream>>>(
    x.data_ptr<at::BFloat16>(),x.stride(0),residual.data_ptr<at::BFloat16>(),weight.data_ptr<at::BFloat16>(),
    static_cast<float>(epsilon),x.size(0),4096,phases.data_ptr<int32_t>(),1);
}
'''


def load():
    from torch.utils.cpp_extension import load_inline
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0';os.environ['MAX_JOBS']='1'
    os.environ['PATH']=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH','')
    directory=ROOT/'tools/phase_mask_norm';directory.mkdir(parents=True,exist_ok=True)
    return load_inline(name='slo_phase_mask_norm_v1',cpp_sources=CPP,cuda_sources=CUDA,
        functions=['masked_norm'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],
        build_directory=str(directory),with_cuda=True,verbose=False)


def build():
    import fcntl,torch
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if not resources().processes():resources().restore(list(range(8)))
        start=time.time();load()
        write(ROOT/'analysis/phase_mask_norm_build.json',dict(time=time.time(),elapsed_s=time.time()-start,
            source_SHA256=digest(__file__),original_body_SHA256=digest(SOURCE/'exact_phase_norm.py'),
            CUDA_context_initialized=torch.cuda.is_initialized(),evidence='CPU_BUILD_ONLY',
            scope='Compile the unchanged scalar reduction body with an all-thread phase predicate. Original256/1024 launch widths remain separate. No GEMM, Attention or precision change.'))
        print('PHASE_MASK_NORM_BUILT',time.time()-start,flush=True)


if __name__=='__main__':build()
