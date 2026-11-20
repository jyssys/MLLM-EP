import unittest
import numpy as np
from .service_model import fit,predict,error_summary


class ServiceModelTests(unittest.TestCase):
    def test_nonnegative_polynomial_recovers_known_curve(self):
        lengths=np.linspace(1000,39000,20);values=3+2*(lengths/8192)+4*(lengths/8192)**2
        beta=fit(lengths,values)
        self.assertTrue(all(x>=0 for x in beta))
        for length,value in zip(lengths,values):self.assertAlmostEqual(predict(beta,length),value,places=7)

    def test_errors_keep_upper_underprediction_distinct(self):
        result=error_summary([10.,30.],[20.,20.])
        self.assertEqual(result['underprediction_frequency'],.5)
        self.assertEqual(result['signed_bias_prediction_minus_actual_ms'],0.)
        self.assertEqual(result['MAE_ms'],10.)
