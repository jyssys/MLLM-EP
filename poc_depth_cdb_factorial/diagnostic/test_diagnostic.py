"""CPU-only checks. Synthetic fixtures are not research measurements."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('diagnostic', Path(__file__).with_name('analyze_diagnostic.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def request(rid, values):
    return dict(id=rid, depths=[dict(depth=d, position=i) for i, d in enumerate(values)])


class Checks(unittest.TestCase):
    def test_population_variance(self):
        self.assertEqual(M.dist([2, 3, 4])['variance'], 2 / 3)

    def test_linear_row_work_does_not_determine_wave_count(self):
        # Equal total work 12. Different temporal max-depth sum: 8 vs 6.
        a = [request('a', [4, 2]), request('b', [2, 4])]
        b = [request('a', [4, 2]), request('b', [4, 2])]
        fa, fb = M.wave_features(M.rank_wave(a)), M.wave_features(M.rank_wave(b))
        self.assertEqual(fa[1], fb[1])
        self.assertEqual(fa[0], 8)
        self.assertEqual(fb[0], 6)

    def test_real_bucket_12_is_preserved(self):
        active = np.array([[9, 10, 11, 12, 13]])
        self.assertEqual(M.wave_features(active)[2], 64)

    def test_lpt_holds_capacity(self):
        ids = list(map(str, range(64)))
        result = M.lpt(ids, {rid: 1 + int(rid) for rid in ids})
        self.assertEqual([list(result.values()).count(i) for i in range(4)], [16] * 4)

    def test_native_wave_identity_from_archived_trace(self):
        _, ranks, _ = M.case(64, 'A1', trace=True)
        for raw in ranks:
            core = [s for s in raw['stages'] if s['stage'] == 'compute_recurrent_batch']
            active = M.rank_wave(raw['outputs'])
            self.assertEqual(active.sum(), sum(s['batch'] for s in core))
            self.assertEqual(int(M.wave_features(active)[0]), len(core))

    def test_all_C64_direct_lengths_match_but_outputs_do_not(self):
        _, _, a = M.case(64, 'A1')
        _, _, b = M.case(64, 'A2')
        p, _ = M.parity(a, b, 'test')
        self.assertEqual((p['length_exact'], p['token_exact'], p['depth_exact']), (64, 44, 43))

    def test_trace_manifest_inputs_unchanged(self):
        import hashlib
        for path, expected in M.read(M.DEST / 'input_manifest.json')['sha256'].items():
            self.assertEqual(hashlib.sha256((M.ROOT.parent / path).read_bytes()).hexdigest(), expected, path)


if __name__ == '__main__':
    unittest.main()
