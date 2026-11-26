"""Score frozen benchmark answers with source-pinned local evaluation."""
import ast
import hashlib
import logging
from pathlib import Path
import random
import re
from typing import Dict
import unicodedata

import numpy as np

from ..mllm_epd_prefill_heavy_token_split import quality_audit

PINNED = {
    "mmmu": (
        "https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/b26bdefc7e5e007962fde8f93b2bb53c99c8d13d/mmmu/utils/eval_utils.py",
        "cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367",
    ),
    "mathvista": (
        "https://raw.githubusercontent.com/lupantech/MathVista/24b1478132e9bb6d32880ebf68ce57b7a0df92b5/evaluation/calculate_score.py",
        "abcd0ff68ec3ed76c3f343be4d01d0f4b88238b660a64e6b998b5f15e3ef56d1",
    ),
}


def source_functions(directory, task, names, variables):
    path = Path(directory) / f"{task}.py"
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != PINNED[task][1]:
        raise ValueError(f"Scorer digest mismatch for {task}")
    nodes = [node for node in ast.parse(content).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Scorer dependency absent for {task}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), variables)
    return variables


def edit_distance(first, second):
    """Same Levenshtein metric as the optional external library."""
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for col, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + int(left != right)))
        previous = current
    return previous[-1]


def local_final_answer(text):
    """Read simple explicit final markers. The input is never interpreted as code."""
    labels = list(re.finditer(r"(?im)^\s*(?:[*# ]*)?(?:final answer|answer)\s*[：:]\s*(.+)$", text))
    if labels:
        text = labels[-1].group(1)
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxes:
        text = boxes[-1]
    return text.strip().strip("*`").strip()


def lexical(text):
    normalized = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    return re.sub(r"\s+", "", normalized.rstrip("."))


class Scorer:
    def __init__(self, asset_repo, scorer_dir):
        quality_audit.ROOT = asset_repo / "poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124"
        self.main = quality_audit.scorers()
        self.mmmu = source_functions(
            scorer_dir, "mmmu",
            {"parse_multi_choice_response", "parse_open_response", "normalize_str",
             "check_is_number", "extract_numbers", "eval_multi_choice", "eval_open"},
            {"re": re, "random": random.Random(42), "np": np, "Dict": Dict},
        )
        mathvista_source = (Path(scorer_dir) / "mathvista.py").read_text()
        tree = ast.parse(mathvista_source)
        pattern, = [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(name, ast.Name) and name.id == "NON_ANSWER_PATTERN"
                            for name in node.targets)]
        functions = source_functions(
            scorer_dir, "mathvista",
            {"is_non_answer", "get_most_similar", "normalize_extracted_answer", "safe_equal"},
            {"re": re, "logging": logging, "distance": edit_distance},
        )
        exec(compile(ast.Module(body=[pattern], type_ignores=[]), "<pinned MathVista pattern>", "exec"),
             functions)
        self.mathvista = functions

    @property
    def provenance(self):
        return {"Main100": self.main[3], "new": PINNED}

    def __call__(self, row, prediction):
        task = row["study_source"]
        refs = list(map(str, row["reference_answers"]))
        options = row["options"] or []
        if task == "Main100":
            return quality_audit.score_request(row, prediction, self.main)
        if task == "MMMU":
            if row["metadata"]["question_type"] == "multiple-choice":
                choices = [chr(ord("A") + index) for index in range(len(options))]
                # Paired model outputs get the same seeded fallback for a request.
                seed = hashlib.sha256(row["request_id"].encode()).digest()
                self.mmmu["random"].seed(seed)
                parsed = self.mmmu["parse_multi_choice_response"](prediction, choices, dict(zip(choices, options)))
                correct = self.mmmu["eval_multi_choice"](refs, parsed)
            else:
                parsed = self.mmmu["parse_open_response"](prediction)
                correct = self.mmmu["eval_open"](refs, parsed)
            return {"score": float(correct), "parsed_answer": parsed, "metric": "MMMU accuracy",
                    "complete_task_scorer": True,
                    "fallback_seed": "Stable per-request seed with the upstream random-guess fallback"}
        extracted = local_final_answer(prediction)
        if task == "MathVista":
            kind = row["metadata"]["question_type"]
            parsed = self.mathvista["normalize_extracted_answer"](
                extracted, options, kind, row["metadata"]["answer_type"], row["metadata"].get("precision"),
                ignore_empty_extractions=True,
            )
            correct = self.mathvista["safe_equal"](parsed, refs[0])
            return {"score": float(correct), "parsed_answer": parsed,
                    "metric": "MathVista upstream normalization after local final-answer extraction",
                    "complete_task_scorer": False, "official_full_pipeline": "NOT_RUN"}
        if task == "MMReason":
            correct = bool(extracted) and lexical(extracted) == lexical(refs[0])
            return {"score": float(correct), "parsed_answer": extracted,
                    "metric": "MMReason local normalized final-answer exact match",
                    "complete_task_scorer": False, "official_full_pipeline": "NOT_RUN"}
        raise ValueError(task)
