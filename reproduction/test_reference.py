"""Fast non-GPU regression probes for reference-output scoring seams."""

import unittest

import torch

from run_reference import normalize_generated_ids, parse_gsm8k
from score_humaneval import candidate_from_output, sandbox_check
from score_reference import numeric_equal, visible_final


class ReferenceOutputTests(unittest.TestCase):
    def test_official_early_eos_prefix_is_stripped(self):
        prompt = torch.tensor([[10, 11, 12]])
        output = torch.tensor([[10, 11, 12, 18, 156892]])
        self.assertEqual(normalize_generated_ids(output, prompt), ([18, 156892], True))

    def test_ordinary_generated_only_return_is_preserved(self):
        prompt = torch.tensor([[10, 11, 12]])
        output = torch.tensor([[18, 156892]])
        self.assertEqual(normalize_generated_ids(output, prompt), ([18, 156892], False))

    def test_parser_does_not_accept_bare_commas_as_numbers(self):
        self.assertEqual(parse_gsm8k("Hello, my friend."), (None, "none"))
        self.assertEqual(parse_gsm8k("The answer is 70,000." )[0], "70000")

    def test_numeric_equivalence_is_reported_separately_from_string_score(self):
        self.assertTrue(numeric_equal("50.0", "50"))
        self.assertFalse(numeric_equal("5", "50"))

    def test_visible_final_ignores_reasoning_number_after_answer_prefix(self):
        self.assertIsNone(visible_final("Answer: Each year the tree has 7 lemons. Continue calculating."))
        self.assertIsNone(visible_final("Answer:\n7 lemons grow the first year."))
        self.assertEqual(visible_final("Answer: 18."), "18")

    def test_humaneval_complete_function_is_not_appended_twice(self):
        record = {"postprocessed_generation": "```python\ndef f(x):\n    return x + 1\n```",
                  "entry_point": "f", "original_prompt": "def f(x):\n    \"\"\"doc\"\"\"\n"}
        self.assertEqual(candidate_from_output(record), ("def f(x):\n    return x + 1", "complete-function"))

    def test_humaneval_complete_function_keeps_prompt_imports(self):
        record = {"postprocessed_generation": "```python\ndef f(xs: List[int]):\n    return len(xs)\n```",
                  "entry_point": "f",
                  "original_prompt": "from typing import List\n\n\ndef f(xs: List[int]):\n    \"\"\"doc\"\"\"\n",
                  "test": "def check(candidate):\n    assert candidate([1, 2]) == 2"}
        candidate, mode = candidate_from_output(record)
        self.assertEqual(mode, "complete-function")
        self.assertEqual(candidate, "from typing import List\n\n\ndef f(xs: List[int]):\n    return len(xs)")
        self.assertTrue(sandbox_check(record)["passed"])

    def test_humaneval_uses_solution_fence_not_first_example_fence(self):
        record = {"postprocessed_generation": "Example:\n```python\nx = [1, 2]\n```\nSolution:\n```python\ndef f(x):\n    return x + 1\n```",
                  "entry_point": "f", "original_prompt": "def f(x):\n    \"\"\"doc\"\"\"\n",
                  "test": "def check(candidate):\n    assert candidate(2) == 3"}
        self.assertEqual(candidate_from_output(record), ("def f(x):\n    return x + 1", "complete-function"))
        self.assertTrue(sandbox_check(record)["passed"])

    def test_humaneval_untrusted_code_runs_in_sandbox(self):
        record = {"postprocessed_generation": "def f(x):\n    return x + 1",
                  "entry_point": "f", "original_prompt": "def f(x):\n    \"\"\"doc\"\"\"\n",
                  "test": "def check(candidate):\n    assert candidate(3) == 4"}
        self.assertTrue(sandbox_check(record)["passed"])

    def test_humaneval_sandbox_cannot_read_host_home(self):
        record = {"postprocessed_generation": "def f(x):\n    return open('/home/esjung/.bashrc').read()",
                  "entry_point": "f", "original_prompt": "def f(x):\n    \"\"\"doc\"\"\"\n",
                  "test": "def check(candidate):\n    candidate(1)"}
        result = sandbox_check(record)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "FileNotFoundError")

    def test_humaneval_sandbox_cannot_escape_through_proc_root(self):
        record = {"postprocessed_generation": "def f(x):\n    return open('/proc/1/root/home/esjung/.bashrc').read()",
                  "entry_point": "f", "original_prompt": "",
                  "test": "def check(candidate):\n    candidate(1)"}
        result = sandbox_check(record)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "PermissionError")


if __name__ == "__main__":
    unittest.main()
