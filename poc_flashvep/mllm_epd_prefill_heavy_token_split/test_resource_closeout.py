"""Idle means no pending stage or active HTTP stream, not low utilization."""
import unittest

from .resource_closeout import pending_proxy_work


class ProxyCloseoutTest(unittest.TestCase):
    def test_pending_encode_or_decode_blocks_idle_closeout(self):
        self.assertEqual(pending_proxy_work(dict(encoder_pending=1,active_http_streams={})),1)
        self.assertEqual(pending_proxy_work(dict(decode_pending=1,active_http_streams={'D':1})),2)

    def test_all_zero_stage_state_is_idle_in_HTTP_scope_only(self):
        self.assertEqual(pending_proxy_work(dict(encoder_pending=0,prefill_pending=0,
            decode_pending=0,active_http_streams={'D':0})),0)


if __name__=='__main__':unittest.main()
