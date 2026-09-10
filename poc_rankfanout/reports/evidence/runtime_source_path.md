# Runtime path source audit

- repository HEAD: `e228b44cec1f5ffa32953e52540086423f84f33f`
- origin HEAD: `e228b44cec1f5ffa32953e52540086423f84f33f`
- vLLM: `0.20.0` at `/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm/__init__.py`
- DeepEP: distribution `1.2.1+73b6ea4`, SM90 extension loaded from `/home/esjung/.venvs/flashvep-deepep-v020/lib/python3.12/site-packages/deep_ep/__init__.py`
- torch/CUDA/NCCL package: `2.11.0+cu129` / `12.9` / `2.28.9`
- AGRS prepare dispatches hidden/top-k tensors with `all_gatherv`; finalize calls `reduce_scatterv`.
- DeepEP HT prepare uses `get_dispatch_layout` and `Buffer.dispatch`; finalize uses `Buffer.combine`.
- This is source evidence. Profiling runs must still prove these paths execute.
