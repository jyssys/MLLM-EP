# Environment Setup

This repository's Phase 1 logic is CPU-only.

Conda is not installed in the current container, so the environment was
specified here and tests were run with the system Python. On a machine with
conda, create the environment with:

```bash
cd /home/work/euisoo.jung/mllm-moe-ep
conda env create -f env/environment.yml
conda activate mllm-moe-ep
```

If a previous attempt failed during the pip step, remove the partial
environment first:

```bash
conda env remove -n mllm-moe-ep
cd /home/work/euisoo.jung/mllm-moe-ep
conda env create -f env/environment.yml
conda activate mllm-moe-ep
```

The lockfile uses PyTorch's CPU wheel index for `torch==2.12.1+cpu` and
`torchvision==0.27.1+cpu`. This does not require a GPU.

If the lockfile needs to be regenerated:

```bash
cd /home/work/euisoo.jung/mllm-moe-ep
uv pip compile env/requirements.txt --torch-backend cpu -o env/requirements.lock
```

If you install manually inside an already-created conda environment, keep the
same CPU wheel source:

```bash
python -m pip install -r env/requirements.lock
```

DeepSpeed is included only so imports are available. Do not run DeepSpeed-MoE
or GPU kernels in Phase 1.
