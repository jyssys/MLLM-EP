"""Reuse the original scalar fused normalization with an explicit launch width.

The kernel body follows the local vLLM scalar width0 implementation.
Only the host launch selects256 threads for short Prefill rows.
Decode retains its original implementation. No GEMM or Attention changes.
The independent exactness gate must pass before a serving run uses this.
"""
import os
from .common import *

CPP=r'''
#include <torch/extension.h>
void phase_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double epsilon);
'''

CUDA=r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#if CUB_VERSION >= 200800
#include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
#else
using CubAddOp = cub::Sum;
#endif

template<typename scalar_t>
__global__ void original_scalar_phase_norm(scalar_t* __restrict__ input,
    const int64_t input_stride, scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight, const float epsilon,
    const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float w = (float)weight[idx];
    input[blockIdx.x * input_stride + idx] = (scalar_t)(x * s_variance * w);
  }
}

void phase_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double epsilon) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda());
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && residual.scalar_type()==at::kBFloat16 && weight.scalar_type()==at::kBFloat16);
  TORCH_CHECK(x.dim()==2 && x.size(1)==4096 && x.size(0)>0 && x.size(0)<256);
  TORCH_CHECK(x.sizes()==residual.sizes() && residual.is_contiguous() && weight.is_contiguous() && weight.numel()==4096);
  const c10::cuda::CUDAGuard guard(x.device());
  original_scalar_phase_norm<at::BFloat16><<<x.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(
    x.data_ptr<at::BFloat16>(),x.stride(0),residual.data_ptr<at::BFloat16>(),weight.data_ptr<at::BFloat16>(),
    static_cast<float>(epsilon),x.size(0),4096);
}
'''


def load():
    from torch.utils.cpp_extension import load_inline
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0';os.environ['MAX_JOBS']='1'
    os.environ['PATH']=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH','')
    directory=ROOT/'tools/exact_phase_norm';directory.mkdir(parents=True,exist_ok=True)
    return load_inline(name='slo_exact_phase_norm_v1',cpp_sources=CPP,cuda_sources=CUDA,
        functions=['phase_norm'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],
        build_directory=str(directory),with_cuda=True,verbose=False)


def build():
    import torch
    started=time.time();module=load()
    original=Path('/home/esjung/external/ouro-unified-planning-vllm026/csrc/libtorch_stable/layernorm_kernels.cu')
    write(ROOT/'analysis/exact_phase_norm_build.json',dict(time=time.time(),start=started,
        torch=torch.__version__,model_revision=MODEL_REVISION,original_source=str(original),original_source_SHA256=digest(original),
        candidate_source_SHA256=digest(__file__),elapsed_seconds=time.time()-started,
        CUDA_arch='9.0',compile_jobs=1,device_context_initialized=torch.cuda.is_initialized(),
        scope='CPU compilation only. No correctness or Serving result. Small Prefill uses the same scalar reduction body with256 launch threads.'))
    print('EXACT_PHASE_NORM_BUILT',time.time()-started)


if __name__=='__main__':build()
