import unittest
from .scheduler_contract import check_contract


class SchedulerContractTests(unittest.TestCase):
    def test_rejects_unused_probe_hook(self):
        with self.assertRaises(ValueError):
            check_contract(dict(D_local_probe_output_pacing=True, wavefront=False))

    def test_accepts_declared_native_configuration(self):
        check_contract(dict(D_local_probe_output_pacing=False, wavefront=False))


if __name__ == '__main__':
    unittest.main()
