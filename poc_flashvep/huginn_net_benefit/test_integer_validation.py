"""A fast exact-int check must preserve every rejection and fallback."""
import unittest
from .integer_validation import exact_int_valid


class IntegerValidationTests(unittest.TestCase):
    def test_native_valid_range(self):
        self.assertTrue(exact_int_valid(0, 32))
        self.assertTrue(exact_int_valid(31, 32))

    def test_outside_range_uses_native_error(self):
        self.assertFalse(exact_int_valid(-1, 32))
        self.assertFalse(exact_int_valid(32, 32))

    def test_bool_does_not_bypass_native_rejection(self):
        self.assertFalse(exact_int_valid(True, 32))

    def test_float_does_not_bypass_native_rejection(self):
        self.assertFalse(exact_int_valid(1., 32))

    def test_integer_subclasses_use_native_validation(self):
        class Integer(int):
            pass
        self.assertFalse(exact_int_valid(Integer(1), 32))


if __name__ == '__main__':
    unittest.main()
