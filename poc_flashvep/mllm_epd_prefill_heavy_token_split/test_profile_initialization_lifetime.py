"""Use the actual worker RPC to retain its one-capture-per-initialization contract."""
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch


class ProfileInitializationTest(unittest.TestCase):
    def test_closed_collector_does_not_authorize_reusing_the_same_worker_for_a_new_capture(self):
        from . import worker_observer as module
        worker=SimpleNamespace(model_runner=object())
        fake=SimpleNamespace(begin=lambda:{'started':True},close=lambda:{'closed':True})
        with patch.dict(os.environ,MLLM_EPD_ROLE='encoder'),patch.object(module,'Collector',return_value=fake):
            self.assertTrue(module.CampaignWorkerExtension.campaign_profile_begin(worker,'first')['started'])
            self.assertTrue(module.CampaignWorkerExtension.campaign_profile_end(worker)['closed'])
            with self.assertRaisesRegex(AssertionError,'One profile per independent worker start'):
                module.CampaignWorkerExtension.campaign_profile_begin(worker,'second')
        fresh=SimpleNamespace(model_runner=object())
        with patch.dict(os.environ,MLLM_EPD_ROLE='encoder'),patch.object(module,'Collector',return_value=fake):
            self.assertTrue(module.CampaignWorkerExtension.campaign_profile_begin(fresh,'second')['started'])


if __name__=='__main__':unittest.main()
