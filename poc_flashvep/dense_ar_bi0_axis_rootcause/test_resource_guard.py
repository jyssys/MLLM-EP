import unittest
from .resource_guard import supervisor_record


class ResourceGuardTest(unittest.TestCase):
    def test_closeout_metadata_is_not_a_supervisor_record(self):
        self.assertFalse(supervisor_record(dict(all8_running=True,active_GPU_processes=[])))
        self.assertFalse(supervisor_record(dict(GPUs=[0,1])))

    def test_actual_supervisor_record(self):
        self.assertTrue(supervisor_record(dict(GPUs=[0,1],supervisor=123)))


if __name__=='__main__':unittest.main()
