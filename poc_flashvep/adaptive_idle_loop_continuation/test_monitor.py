"""Exercise partial burner restoration without starting GPU processes."""
import unittest
from unittest.mock import patch
from .monitor import restore_missing_burners


class Owner:
    def __init__(self,active,owned):
        self.active=active;self.owned=owned;self.restored=[]
    def processes(self):return self.active
    def burners(self):return self.owned
    def restore(self,gpus):self.restored.extend(gpus)


class RestorationTests(unittest.TestCase):
    def test_seven_burners_retry_only_missing_gpu(self):
        owner=Owner([dict(gpu=g,pid=100+g) for g in range(1,8)],
                    {g:200+g for g in range(1,8)})
        with patch('os.getsid',side_effect=lambda pid:pid+100):
            self.assertTrue(restore_missing_burners(owner))
        self.assertEqual(owner.restored,[0])

    def test_unrelated_job_prevents_restoration(self):
        owner=Owner([dict(gpu=0,pid=100)],{})
        with patch('os.getsid',return_value=100):
            self.assertFalse(restore_missing_burners(owner))
        self.assertEqual(owner.restored,[])

    def test_full_occupancy_does_not_restart(self):
        owner=Owner([dict(gpu=g,pid=100+g) for g in range(8)],
                    {g:200+g for g in range(8)})
        with patch('os.getsid',side_effect=lambda pid:pid+100):
            self.assertFalse(restore_missing_burners(owner))
        self.assertEqual(owner.restored,[])


if __name__=='__main__':unittest.main()
