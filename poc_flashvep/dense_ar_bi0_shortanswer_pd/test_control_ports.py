import threading
import unittest
from .control_ports import candidate_ports,call_listener


class ControlPortTests(unittest.TestCase):
    def test_candidates_exclude_ephemeral_ports(self):
        values=candidate_ports((32768,60999))
        self.assertTrue(values)
        self.assertTrue(all(not 32768<=p<=60999 for p in values))
        self.assertEqual(len(values),len(set(values)))

    def test_listener_error_releases_readiness_but_records_failure(self):
        errors=[];ready=threading.Event()
        def failed(*args):raise OSError('Address already in use')
        call_listener(failed,errors,{},ready,None,'127.0.0.1',20000)
        self.assertTrue(ready.is_set())
        self.assertEqual(len(errors),1)
        self.assertIn('Address already in use',errors[0])
