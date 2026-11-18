"""Check the exact Token PrefixChannel Client replacement seam."""
from types import SimpleNamespace
import unittest


class CohortClientAliasTests(unittest.TestCase):
    def test_token_installer_captures_the_registered_barrier_client(self):
        from .fixed_matched_cohort_v3 import preserve_token_client
        raw=lambda:'raw'
        registered=lambda:('registered',raw())
        worker=SimpleNamespace(Client=registered)
        atomic=SimpleNamespace(Client=raw)
        preserve_token_client(dict(token_axis=True),'decode',worker,atomic)
        # token_axis.install captures ar_worker.Client, then replaces both.
        captured=atomic.Client
        token_connected=lambda:('prefix',captured())
        worker.Client=token_connected
        self.assertEqual(worker.Client(),('prefix',('registered','raw')))

    def test_source_prefix_channel_is_not_replaced(self):
        from .fixed_matched_cohort_v3 import preserve_token_client
        original=lambda:'raw'
        worker=SimpleNamespace(Client=lambda:'registered')
        atomic=SimpleNamespace(Client=original)
        preserve_token_client(dict(token_axis=True),'prefill',worker,atomic)
        self.assertIs(atomic.Client,original)


if __name__=='__main__':
    unittest.main()
