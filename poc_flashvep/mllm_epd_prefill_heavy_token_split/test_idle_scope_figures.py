"""Runtime labels and physical activity cannot erase scope or observer failures."""
import copy
import unittest

from .idle_scope_figures import runtime_fingerprint,activity_rows,CORE_FILES


class IdleScopeTest(unittest.TestCase):
    def config(self):
        return dict(manifest_SHA256='main',reference_SHA256='replay',servers=dict(E=dict(role='encoder',gpu=4,
            command=['python','--max-num-batched-tokens','114688','--mm-processor-cache-gb','0.0',
                '--no-async-scheduling'],source_SHA256={name:'abc' for name in CORE_FILES})))

    def test_source_and_cache_policy_changes_split_fingerprints(self):
        original=self.config();fingerprint,_=runtime_fingerprint(original)
        changed=copy.deepcopy(original);changed['servers']['E']['source_SHA256']['proxy.py']='new'
        self.assertNotEqual(fingerprint,runtime_fingerprint(changed)[0])
        changed=copy.deepcopy(original);changed['servers']['E']['command'][-2]='1.0'
        self.assertNotEqual(fingerprint,runtime_fingerprint(changed)[0])

    def test_QPS_and_names_are_not_runtime_equivalence_inputs(self):
        original=self.config();fingerprint,_=runtime_fingerprint(original)
        original['QPS']=3;original['servers']['E']['name']='other'
        self.assertEqual(fingerprint,runtime_fingerprint(original)[0])

    def test_failed_observer_keeps_diagnostic_and_short_roi(self):
        stage=dict(kernel_busy_fraction=.8,kernel_idle_fraction=.2,scheduler_waiting_mean_per_observation=2,
            scheduler_waiting_max=3,actual_scheduled_requests_max=4)
        roi=dict(duration_seconds=10,roles=dict(encoder=stage),E_D_simultaneous_kernel_idle_fraction=.1,
            E_D_simultaneous_idle_during_P_busy_fraction_total=.05,E_D_simultaneous_idle_fraction_of_P_busy=.0625)
        data=dict(QPS=3,frozen_completion_roi=[30,70],observer_gate_passed=False,ROIs=dict(central_arrival_window=roi))
        row=activity_rows('short',data,dict(observer_gate_passed=False),'fingerprint')[0]
        self.assertEqual(row['evidence'],'DIAGNOSTIC')
        self.assertEqual(row['ROI'],'central_arrival_window_observed_intersection')
        self.assertEqual(row['SM_active'],'NOT_MEASURED')
        self.assertEqual(row['simultaneous_idle_percent_of_P_busy'],6.25)


if __name__=='__main__':unittest.main()
