import unittest
from .measurement_export import quantile,queue_diagnostic
from .step_prediction_validation import error_summary


class MeasurementTests(unittest.TestCase):
    def test_quantile_preserves_empty_and_interpolation(self):
        self.assertIsNone(quantile([],.95))
        self.assertAlmostEqual(quantile([1,2,3],.95),2.9)

    def test_overload_keeps_unfinished_arrivals_in_denominator(self):
        rows=[dict(arrival=i/10,finish=1+i/10) for i in range(10)]
        result=queue_diagnostic(rows)
        self.assertEqual(result['status'],'OVERLOAD_DIAGNOSTIC')
        self.assertEqual(result['unfinished_at_last_arrival'],10)
        self.assertAlmostEqual(result['final_Drain_seconds'],1.)

    def test_serial_window_is_not_called_sustainable(self):
        self.assertEqual(queue_diagnostic([dict(arrival=0.,finish=1.)])['status'],
            'NOT_APPLICABLE_SERIAL_OR_ZERO_SPAN')

    def test_error_sign_and_underprediction_are_explicit(self):
        result=error_summary([1.,2.],[.9,2.1])
        self.assertAlmostEqual(result['MAE_ms'],100.)
        self.assertEqual(result['underprediction_frequency'],.5)


if __name__=='__main__':
    unittest.main()
