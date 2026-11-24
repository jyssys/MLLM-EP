"""Recalculate client timestamps without redefining token or SLO contracts."""
import unittest


class BaselineAnalysisTest(unittest.TestCase):
    def row(self,times):
        arrival=1.
        return dict(token_times=times,token_ids=list(range(len(times))),output_tokens=len(times),
            scheduled_arrival=arrival,usage=dict(completion_tokens=len(times)),
            TTFT_ms=(times[0]-arrival)*1000,
            ITL_ms=[(b-a)*1000 for a,b in zip(times,times[1:])],
            TPOT_ms=(times[-1]-times[0])*1000/(len(times)-1) if len(times)>1 else 0)

    def test_single_and_co_delivered_output_tokens_keep_original_definitions(self):
        from .baseline_confirmation_analysis import verify_timing
        verify_timing(self.row([1.01]))
        verify_timing(self.row([1.01,1.02,1.02]))

    def test_wrong_stored_latency_is_rejected(self):
        from .baseline_confirmation_analysis import verify_timing
        row=self.row([1.01,1.02]);row['TTFT_ms']+=1
        with self.assertRaises(AssertionError):verify_timing(row)

    def test_request_count_cannot_create_a_start_level_interval(self):
        from .baseline_confirmation_analysis import bootstrap
        self.assertIsNone(bootstrap([100.]))
        self.assertEqual(bootstrap([100.,100.,100.]),[100.,100.])


if __name__=='__main__':unittest.main()
