from __future__ import annotations

from ..constants import REPORTED_REGISTRY


def registry() -> dict:
    sources = {
        "technical_report": "https://arxiv.org/abs/2511.21631 (Table 3, printed p.16)",
        "official_repo": "https://github.com/QwenLM/Qwen3-VL (Evaluation Reproduction)",
        "vlmevalkit": "https://github.com/open-compass/VLMEvalKit",
    }
    return {
        "model_column": "Qwen3-VL-30B-A3B / Instruct",
        "reported": REPORTED_REGISTRY,
        "sources": sources,
        "protocol_status": {
            "MMStar": "PARTIALLY_DOCUMENTED",
            "ChartQA_TEST": "PARTIALLY_DOCUMENTED",
        },
        "warning": "MMMU-Pro/Thinking/235B and third-party vanilla values are not substitutes",
    }

