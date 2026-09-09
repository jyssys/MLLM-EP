#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
research_env=/home/esjung/.venvs/scheduling-nanoflow-py310
research_ref=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/refs/Nanoflow-h100
if [[ ! -x "$research_env/bin/python" ]]; then
  /home/esjung/.venvs/libra-supplement-py310/bin/python -m venv "$research_env"
fi
"$research_env/bin/python" -m pip install --upgrade pip 'setuptools==75.8.2' 'cmake==3.31.10' ninja wheel pybind11
"$research_env/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$research_env/bin/python" -m pip install nvtx loguru 'transformers==4.57.3' accelerate matplotlib 'nvmath-python==0.5.0' 'flashinfer-python==0.3.1' gurobipy pandas scipy networkx seaborn safetensors
"$research_env/bin/python" -m pip install -e "$research_ref" --no-deps --no-build-isolation
"$research_env/bin/python" -m pip freeze
