"""Pin the heldout protocol before any performance result exists."""
import unittest


class TokenConfirmationTest(unittest.TestCase):
    def test_three_reproducible_blocks_keep_strong_native_and_both_candidates(self):
        from .token_confirmation_control import METHODS,method_orders
        first=method_orders()
        self.assertEqual(len(first),3)
        self.assertEqual(first,method_orders())
        self.assertTrue(all(sorted(row)==sorted(METHODS) for row in first))
        self.assertNotIn('Static25',METHODS)

    def test_main_and_heldout_use_frozen_ids_without_changing_the_manifest(self):
        from .token_confirmation_control import workload
        from .common import digest
        path,main,heldout=workload()
        self.assertEqual(digest(path),'910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1')
        self.assertEqual(len(main),100)
        self.assertEqual(len(heldout),50)
        self.assertTrue({r['request_id'] for r in heldout}<={r['request_id'] for r in main})


if __name__=='__main__':unittest.main()
