"""P-busy marginals overlap. Only four exclusive states conserve that time."""
import unittest

from .idle_final_summary import P_busy_states


class PBusyStateTest(unittest.TestCase):
    def test_disjoint_states_conserve_P_busy_not_total_ROI(self):
        roi=dict(duration_seconds=10,roles=dict(prefill=dict(kernel_busy_fraction=.4)),
            E_D_simultaneous_idle_during_P_busy_fraction_total=.1,
            P_busy_E_idle_fraction_total=.2,P_busy_D_idle_fraction_total=.25,
            all_kernel_active_fraction=.05)
        rows=P_busy_states(roi);exclusive=[r for r in rows if r['accounting']=='DISJOINT_PARTITION']
        self.assertAlmostEqual(sum(r['fraction_total_ROI'] for r in exclusive),.4)
        self.assertAlmostEqual(sum(r['fraction_of_P_busy'] for r in exclusive),1)
        self.assertEqual(len(rows),6)

    def test_inconsistent_overlap_is_not_silently_normalized(self):
        roi=dict(duration_seconds=10,roles=dict(prefill=dict(kernel_busy_fraction=.4)),
            E_D_simultaneous_idle_during_P_busy_fraction_total=.3,
            P_busy_E_idle_fraction_total=.2,P_busy_D_idle_fraction_total=.25,
            all_kernel_active_fraction=.05)
        with self.assertRaises(AssertionError):P_busy_states(roi)


if __name__=='__main__':unittest.main()
