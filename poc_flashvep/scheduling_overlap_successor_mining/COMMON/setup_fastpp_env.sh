#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
research_env=/home/esjung/.venvs/scheduling-fastpp-py310
research_ref=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/refs/FastPP
if [[ ! -x "$research_env/bin/python" ]]; then
  /home/esjung/.venvs/libra-supplement-py310/bin/python -m venv "$research_env"
fi
"$research_env/bin/python" -m pip install --upgrade pip uv
"$research_env/bin/uv" pip install --python "$research_env/bin/python" \
  -e "$research_ref/python[srt]" -c "$research_ref/constraints.txt" \
  --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
"$research_env/bin/python" -m pip freeze
