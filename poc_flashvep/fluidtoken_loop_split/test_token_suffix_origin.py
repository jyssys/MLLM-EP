"""Execute actual scheduler progress on CPU. No GPU/model qualification."""
import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from .token_suffix_origin import make_task


class Phase(Enum):
    PREFILLING_PRELUDE = 1
    PREFILLING_RECURRENT = 2
    CODA = 3


@dataclass
class Task:
    request: object
    token_start: int
    token_count: int
    depth: int = None
    chunk_start: int = 0
    prefill_hidden_row_offset: int = 0
    completes_prompt: bool = False


def extracted(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec', flags=__import__('__future__').annotations.compiler_flag), namespace)
    return namespace[name]


class TokenSuffixOriginTests(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[2]
        self.original = extracted(repo / 'poc_flashvep/adaptive_recurrence_cut/staged.py', 'task',
            dict(maximum=8192, RequestPhase=Phase, ScheduledRequest=Task))
        self.progress = extracted('/home/esjung/external/looped-pd-abc-rlt/vllm_rlt/core/scheduler.py',
            '_apply_wavefront_prefill_batch_progress', dict(RequestPhase=Phase, PrefillProgress=lambda *a: a))

    def case(self, length, prefix, depth, chunk):
        request = NS(prompt_token_ids=list(range(length)), prefill_window_origin=prefix,
            prefill_reclaimable_tokens=prefix, num_preluded_tokens=length,
            prefill_depth_ranges=[[prefix, length] for _ in range(depth)], num_prefilled_tokens=prefix,
            request_id='imported', loops_done=0)
        scheduler = NS(config=NS(prefill_chunk_size=chunk), total_ut_steps=depth,
            cache_manager=NS(publish_prefix=lambda *a: None), enqueue=lambda *a: None,
            _prompt_destination=lambda r: Phase.CODA, _prelude_window_open=lambda r: False)
        return scheduler, request

    def test_actual_legacy_progress_reproduces_both_failures(self):
        for length, prefix, depth in ((2048, 1984, 32), (4096, 3072, 4)):
            scheduler, request = self.case(length, prefix, depth, length)
            task = self.original(scheduler, request, Phase.PREFILLING_RECURRENT,
                                 length, start=prefix, depth=depth - 1)
            self.assertEqual(task.chunk_start, 0)
            with self.assertRaisesRegex(RuntimeError, 'wavefront completed prompt chunks out of order'):
                self.progress(scheduler, Phase.PREFILLING_RECURRENT, [task])

    def test_candidate_preserves_origin_and_absolute_rows(self):
        for length, prefix, depth in ((2048, 1984, 32), (4096, 3072, 4)):
            scheduler, request = self.case(length, prefix, depth, length)
            task = make_task(8192, Phase, Task)(scheduler, request, Phase.PREFILLING_RECURRENT,
                                               length, start=prefix, depth=depth - 1)
            self.assertEqual(task.chunk_start, prefix)
            self.assertEqual(task.prefill_hidden_row_offset, prefix)
            self.progress(scheduler, Phase.PREFILLING_RECURRENT, [task])
            self.assertEqual(request.prefill_reclaimable_tokens, length)
            self.assertEqual(request.loops_done, depth)

    def test_origin_zero_is_unchanged_and_cut_is_respected(self):
        scheduler, request = self.case(2048, 0, 32, 256)
        for start in (0, 256, 512):
            request.prefill_depth_ranges[0][0] = start
            a = self.original(scheduler, request, Phase.PREFILLING_RECURRENT, 256, start=start, depth=0)
            b = make_task(8192, Phase, Task)(scheduler, request, Phase.PREFILLING_RECURRENT, 256, start=start, depth=0)
            self.assertEqual(a, b)
        request.arc_export_cut = 31
        self.assertIsNone(make_task(8192, Phase, Task)(scheduler, request, Phase.PREFILLING_RECURRENT,
                                                     256, start=512, depth=31))


if __name__ == '__main__':
    unittest.main()
