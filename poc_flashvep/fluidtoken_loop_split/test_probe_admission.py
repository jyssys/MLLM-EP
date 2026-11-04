import unittest
from types import SimpleNamespace as N
from .probe_admission import AdmissionConfig


class ProbeAdmissionTests(unittest.TestCase):
    def test_background_is_not_changed(self):
        requests = {str(i): N(request_id=str(i), phase='RECURRENT') for i in range(3)}
        cfg = AdmissionConfig(dict(D_probe_count=12, D_probe_max_active_decode=2), lambda row: None)
        cfg.engine = N(scheduler=N(requests=requests))
        self.assertEqual(cfg.get('D_probe_count'), 0)
        self.assertEqual(len(requests), 3)
        requests.pop('2')
        self.assertEqual(cfg.get('D_probe_count'), 12)
        self.assertEqual(cfg['D_probe_count'], 12)

    def test_probe_and_prefill_do_not_count_as_decode(self):
        cfg = AdmissionConfig(dict(D_probe_count=12, D_probe_max_active_decode=1), lambda row: None)
        cfg.engine = N(scheduler=N(requests={
            'a': N(request_id='a', phase='RECURRENT'),
            'b': N(request_id='b', phase='PREFILLING_RECURRENT'),
            'c': N(request_id='autonomous_probe_0', phase='RECURRENT')}))
        self.assertEqual(cfg.get('D_probe_count'), 12)

    def test_optional_boundary_phase_guard(self):
        cfg = AdmissionConfig(dict(D_probe_count=12, D_probe_max_active_decode=2,
                                   D_probe_admission_scope='all_decode_phases'), lambda row: None)
        cfg.engine = N(scheduler=N(requests={
            'a': N(request_id='a', phase='RECURRENT'),
            'b': N(request_id='b', phase='PRELUDE'),
            'c': N(request_id='c', phase='CODA')}))
        self.assertEqual(cfg.get('D_probe_count'), 0)
        self.assertEqual(len(cfg.engine.scheduler.requests), 3)
        cfg.engine.scheduler.requests.pop('c')
        self.assertEqual(cfg.get('D_probe_count'), 12)

    def test_paced_probe_has_one_owner(self):
        cfg = AdmissionConfig(dict(D_probe_count=12, D_probe_max_active_decode=4,
                                   D_local_probe_output_pacing=True), lambda row: None)
        requests = {'a': N(request_id='a', phase='RECURRENT'),
                    'p': N(request_id='autonomous_probe_0', phase='PREFILLING_RECURRENT')}
        cfg.engine = N(scheduler=N(requests=requests))
        self.assertEqual(cfg.get('D_probe_count'), 0)
        requests.pop('p')
        self.assertEqual(cfg.get('D_probe_count'), 12)


if __name__ == '__main__':
    unittest.main()
