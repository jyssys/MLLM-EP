#!/usr/bin/env bash
# Command ledger. Review the GPU lease before any GPU command.
# Named runs refuse to overwrite existing evidence. These are recorded invocations, not an automatic rerun.
set -euo pipefail
cd /home/esjung/MLLM-EP-ouro-request-pp

# correctness_P128_C4_v4.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs --correctness-prompt 128 --correctness-count 4 --correctness-depths 4

# correctness_P128_C4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs --correctness-prompt 128 --correctness-count 4 --correctness-depths 4

# correctness_P2048_C1_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs --correctness-prompt 2048 --correctness-count 1 --correctness-depths 4

# correctness_P512_C2_v4.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs --correctness-prompt 512 --correctness-count 2 --correctness-depths 4

# correctness_P512_C2_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs --correctness-prompt 512 --correctness-count 2 --correctness-depths 4

# correctness_graph_v2.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs

# correctness_graph_v3.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs

# correctness_graph_v4.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs

# correctness_graph_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs

# correctness_receive_pool.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs --preallocated-recv --correctness-prompt 128 --correctness-count 4 --correctness-depths 4 --runtime-label receive_pool

# correctness_receive_pool_rope.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness --graphs --preallocated-recv --precompute-rotary --correctness-prompt 128 --correctness-count 4 --correctness-depths 4 --runtime-label receive_pool_rope

# correctness_v1.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode correctness

# depth_trace_layer_pp4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label depth_trace_layer_pp4_v5 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/depth_trace_plan_v5.json --graphs --trace

# depth_trace_loop_pp4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label depth_trace_loop_pp4_v5 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/depth_trace_plan_v5.json --graphs --trace

# eos_layer_pp4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label eos_layer_pp4_v5 --depth 4 --concurrency 4 --prompt-len 128 --output-len 64 --graphs --stop-eos

# eos_loop_pp4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label eos_loop_pp4_v5 --depth 4 --concurrency 4 --prompt-len 128 --output-len 64 --graphs --stop-eos

# observer_layer_pp4_clean_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label observer_layer_pp4_clean_v5 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/observer_plan_v5.json --graphs

# observer_layer_pp4_trace_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label observer_layer_pp4_trace_v5 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/observer_plan_v5.json --graphs --trace

# observer_loop_pp4_clean_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label observer_loop_pp4_clean_v5 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/observer_plan_v5.json --graphs

# observer_loop_pp4_trace_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label observer_loop_pp4_trace_v5 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/observer_plan_v5.json --graphs --trace

# primary_layer_pp4_r0.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label primary_layer_pp4_r0 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/plan_r0.json --graphs

# primary_layer_pp4_v5_r0.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label primary_layer_pp4_v5_r0 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/plan_v5_r0.json --graphs

# primary_layer_pp4_v5_r1.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label primary_layer_pp4_v5_r1 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/plan_v5_r1.json --graphs

# primary_layer_pp4_v5_r2.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label primary_layer_pp4_v5_r2 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/plan_v5_r2.json --graphs

# primary_loop_pp4_v5_r0.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label primary_loop_pp4_v5_r0 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/plan_v5_r0.json --graphs

# primary_loop_pp4_v5_r1.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label primary_loop_pp4_v5_r1 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/plan_v5_r1.json --graphs

# primary_loop_pp4_v5_r2.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label primary_loop_pp4_v5_r2 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/plan_v5_r2.json --graphs

# probe_layer_pp4_v1.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label probe_layer_pp4_v1 --depth 4 --concurrency 4 --prompt-len 128 --output-len 8 --trace --graphs

# probe_layer_pp4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label probe_layer_pp4_v5 --depth 4 --concurrency 4 --prompt-len 128 --output-len 8 --trace --graphs

# probe_loop_pp4_v1.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label probe_loop_pp4_v1 --depth 4 --concurrency 4 --prompt-len 128 --output-len 8 --trace --graphs

# probe_loop_pp4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label probe_loop_pp4_v5 --depth 4 --concurrency 4 --prompt-len 128 --output-len 8 --trace --graphs

# profile_layer_pp4_R2_C8_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/profiles/layer_pp4_R2_C8_v5 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label nsys_layer_pp4_R2_C8_v5 --depth 2 --concurrency 8 --prompt-len 128 --output-len 8 --graphs --trace --nsys

# profile_layer_pp4_R4_C4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/profiles/layer_pp4_R4_C4_v5 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label nsys_layer_pp4_R4_C4_v5 --depth 4 --concurrency 4 --prompt-len 128 --output-len 8 --graphs --trace --nsys

# profile_loop_pp4_R2_C8_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/profiles/loop_pp4_R2_C8_v5 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label nsys_loop_pp4_R2_C8_v5 --depth 2 --concurrency 8 --prompt-len 128 --output-len 8 --graphs --trace --nsys

# profile_loop_pp4_R4_C4_v5.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false -o /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/profiles/loop_pp4_R4_C4_v5 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label nsys_loop_pp4_R4_C4_v5 --depth 4 --concurrency 4 --prompt-len 128 --output-len 8 --graphs --trace --nsys

# receive_pool_layer_pp4.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label receive_pool_layer_pp4 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/receive_pool_plan.json --graphs --preallocated-recv

# receive_pool_loop_pp4.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label receive_pool_loop_pp4 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/receive_pool_plan.json --graphs --preallocated-recv

# receive_pool_rope_layer_pp4.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement layer_pp4 --label receive_pool_rope_layer_pp4 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/receive_pool_rope_plan.json --graphs --preallocated-recv --precompute-rotary

# receive_pool_rope_loop_pp4.json
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 /home/esjung/.venvs/llada2-mini-reference/bin/python -u -m poc_flashvep.ouro_request_pp.benchmark --mode benchmark --placement loop_pp4 --label receive_pool_rope_loop_pp4 --case-file /home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_request_pp_discovery/receive_pool_rope_plan.json --graphs --preallocated-recv --precompute-rotary
