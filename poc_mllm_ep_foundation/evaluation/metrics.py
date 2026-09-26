from __future__ import annotations

import re


def chartqa_relaxed_correct(prediction: str, answer: str) -> bool:
    """Exact port of the frozen lmms-eval ChartQA relaxed scorer."""
    def to_float(text: str):
        try:
            return float(text.rstrip("%")) / 100.0 if text.endswith("%") else float(text)
        except ValueError:
            return None

    pred, target = str(prediction), str(answer)
    pred_value, target_value = to_float(pred), to_float(target)
    if pred_value is not None and target_value:
        return abs(pred_value - target_value) / abs(target_value) <= 0.05
    return pred.lower() == target.lower()


def extract_mcq_letter(text: str) -> str | None:
    """Port of lmms-eval's priority-based MMStar MCQ extraction."""
    if not text or not text.strip():
        return None
    choices = "ABCD"
    value = text.strip().upper()
    for char in ",.!?;:'\"":
        value = value.strip(char)
    padded = f" {value} "
    candidates: list[tuple[str, int, int]] = []
    formats = (("({})", 6), ("{}.", 5), ("{}:", 4), ("{})", 3), ("{} ", 2))
    for pattern, priority in formats:
        for choice in choices:
            token = pattern.format(choice)
            if token in padded:
                candidates.append((choice, padded.rfind(token), priority))
    stripped = padded.strip()
    for choice in choices:
        if stripped.startswith(choice) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((choice, 0, 10))
        if stripped.endswith(choice) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((choice, len(padded) - 1, 9))
    if not candidates:
        for choice in choices:
            match = list(re.finditer(rf"(?<![A-Z]){choice}(?![A-Z])", padded))
            if match:
                candidates.append((choice, match[-1].start(), 0))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    return candidates[0][0]
