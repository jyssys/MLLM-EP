import unittest
from types import SimpleNamespace as S
from .partial_prepare import unused_partial_coda


class PartialPrepareTests(unittest.TestCase):
    def batch(self, loops, depth=0, operation='prelude'):
        return S(operation=S(value=operation), loop_count=loops, spans=[S(depth=depth)])

    def test_full_native_coda_is_retained(self):
        self.assertFalse(unused_partial_coda(self.batch(32), 32))

    def test_partial_native_forward_has_no_coda(self):
        self.assertTrue(unused_partial_coda(self.batch(31), 32))

    def test_token_prefix_at_full_depth_keeps_its_valid_coda(self):
        self.assertFalse(unused_partial_coda(self.batch(32), 32))

    def test_split_recurrent_terminal_path_is_unchanged(self):
        self.assertFalse(unused_partial_coda(self.batch(1, 31, 'recurrent'), 32))

    def test_standalone_prelude_does_not_use_this_rule(self):
        self.assertFalse(unused_partial_coda(self.batch(0), 32))


if __name__ == '__main__':
    unittest.main()
