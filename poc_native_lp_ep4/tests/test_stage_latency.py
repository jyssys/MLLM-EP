import unittest

from poc_native_lp_ep4.scripts.stage_latency import align_completion, summarize_stages


class StageLatencyTest(unittest.TestCase):
    def test_alignment_is_actual_not_interpolation(self):
        index, error = align_completion(201000000, [100000000, 200000000, 300000000])
        self.assertEqual(index, 1)
        self.assertEqual(error, 1)
        with self.assertRaises(ValueError):
            align_completion(250000000, [100000000, 200000000, 300000000])

    def test_wave_is_the_sampling_unit(self):
        rows = []
        for wave, stages in ((1, [100, 110, 120]), (2, [200])):
            for value in stages:
                rows.append({'run': 'x', 'wave': wave, 'phase': 'LATER_PREFILL_STAGE',
                    'arm': 'LP_G8', 'modality': 'VISUAL_PREFILL', 'repeat': wave,
                    'victim_mean_itl_ms': value, 'victim_max_itl_ms': value,
                    'victim_max_vsel_ms': None})
        waves, summary = summarize_stages(rows)
        self.assertEqual(len(waves), 2)
        self.assertEqual(summary[0]['n_waves'], 2)
        self.assertEqual(summary[0]['mean_stage_itl_ms']['mean'], 155)


if __name__ == '__main__':
    unittest.main()
