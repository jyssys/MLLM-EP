#!/usr/bin/env bash
set -euo pipefail
cd /home/esjung/MLLM-EP-github
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.diagnostics
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.oracles
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.characterize
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.finalize
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.integrity
