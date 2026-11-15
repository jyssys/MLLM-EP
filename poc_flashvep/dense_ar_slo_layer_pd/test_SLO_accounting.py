"""Apply the frozen threshold to each actual request output cap."""
import unittest
from .analyze import request_SLOs


class MixedSLOTests(unittest.TestCase):
    def test_each_request_uses_its_own_cap(self):
        cfg=dict(request_caps=[16,64])
        rows=[dict(sample_id=i,ttft_ms=100.,tpot_ms=20.,output_tokens=cap)
            for i,cap in enumerate(cfg['request_caps'])]
        limits={str(cap):{k:dict(TTFT_ms=1000.,TPOT_ms=25.,ITL_ms=30. if cap==16 else 31.)
            for k in ('TTFT_priority','Balanced','Decode_priority')} for cap in (16,64)}
        result=request_SLOs(cfg,rows,[[30.5],[30.5]],limits,10.)['Decode_priority']
        self.assertEqual(result['Joint_attainment'],.5)
        self.assertEqual(result['Output_token_SLO_goodput'],6.4)
        self.assertEqual(result['ITL_threshold_exceedances'],1)
        self.assertIsNone(result['percentile_Decode_absolute_pass'])


if __name__=='__main__':unittest.main()
