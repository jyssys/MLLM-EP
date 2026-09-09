#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/esjung/.venvs/scheduling-nanoflow-py310/bin:/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
research_ref=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/refs/Nanoflow-h100
research_prefix=/home/esjung/.venvs/scheduling-nanoflow-native
research_site=/home/esjung/.venvs/scheduling-nanoflow-py310/lib/python3.10/site-packages
cmake -S "$research_ref/nanoflow/pybind" -B "$research_ref/nanoflow/pybind/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=/home/esjung/.venvs/scheduling-nanoflow-py310/bin/python \
  -DPYTHON_EXECUTABLE=/home/esjung/.venvs/scheduling-nanoflow-py310/bin/python \
  -DPython_ROOT_DIR=/home/esjung/.venvs/scheduling-nanoflow-py310 \
  -DCMAKE_PREFIX_PATH="$research_prefix;$research_site/pybind11/share/cmake/pybind11" \
  -DCMAKE_CXX_FLAGS="-I$research_prefix/include -I$research_site/nvidia/nccl/include" \
  -DCMAKE_CUDA_FLAGS="-I$research_prefix/include -I$research_site/nvidia/nccl/include" \
  -DCMAKE_LIBRARY_PATH="$research_prefix/lib;$research_site/nvidia/nccl/lib" \
  -DCMAKE_BUILD_RPATH="$research_prefix/lib;$research_site/nvidia/nccl/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$research_site/nvidia/nccl/lib -l:libnccl.so.2"
cmake --build "$research_ref/nanoflow/pybind/build" -j 8
python -c 'from nanoflow.models.qwen2_moe.qwen2_moe_ep import Pipeline; print("NANOFLOW_NATIVE_IMPORT_PASS")'
