from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL = Path(
    "/home/esjung/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    + MODEL_REVISION
)
ALLOWED_PHYSICAL_GPUS = (4, 5, 6, 7)
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48
DTYPE_BYTES = 2

EXPECTED_TEXT_CONFIG = {
    "hidden_size": HIDDEN_SIZE,
    "moe_intermediate_size": INTERMEDIATE_SIZE,
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": TOP_K,
    "num_hidden_layers": NUM_LAYERS,
}

OFFICIAL_GENERATION = {
    "do_sample": True,
    "seed": 3407,
    "temperature": 0.7,
    "top_p": 0.8,
    "sampling_top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "max_new_tokens": 32768,
    "stop": "checkpoint EOS IDs",
}

REPORTED_REGISTRY = {
    "MMStar": {"score": 72.1, "metric": "accuracy"},
    "ChartQA_TEST": {"score": 86.8, "metric": "relaxed accuracy"},
    "RealWorldQA": {"score": 73.7, "metric": "accuracy"},
    "MMMU": {"score": 74.2, "metric": "accuracy"},
    "OCRBench": {"score": 903, "metric": "raw score"},
    "DocVQA_TEST": {"score": 95.0, "metric": "ANLS"},
}

HISTORICAL_ROUTE_TRACE = REPO / (
    "poc_flashvep/deepep_revalidation/results/"
    "prerouter_visual_signal_20260821_133700"
)
BOUNDED_HIDDEN_TRACE = REPO / (
    "poc_spatial_basis_ep/artifacts/20260924_spatial_basis_ep/discovery_trace"
)

