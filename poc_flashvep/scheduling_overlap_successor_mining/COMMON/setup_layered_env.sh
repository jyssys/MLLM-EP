#!/usr/bin/env bash
set -euo pipefail
# Isolated dependency setup only; no benchmark, no baseline-env mutation.
export CUDA_VISIBLE_DEVICES=4,5,6,7
research_env=/home/esjung/.venvs/scheduling-layered-py310
research_ref=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/refs/layered-prefill
if [[ ! -x "$research_env/bin/python" ]]; then
  /home/esjung/.venvs/libra-supplement-py310/bin/python -m venv "$research_env"
fi
"$research_env/bin/python" -m pip install --upgrade pip
"$research_env/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$research_env/bin/python" -m pip install -r "$research_ref/requirements.txt" ninja cmake wheel packaging setuptools pybind11 uv
# Python 3.10's bundled setuptools 59 lacks PEP 660; CMake 4 removed FindCUDA.
"$research_env/bin/python" -m pip install --upgrade 'setuptools==75.8.2' 'cmake==3.31.10'
"$research_env/bin/python" -m pip freeze
