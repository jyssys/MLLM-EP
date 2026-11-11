"""Separate streams must not silently route a mixed dependency batch."""
import unittest
from .stream_route import prompt_route


class StreamRouteTests(unittest.TestCase):
    def test_decode_batch_retains_its_route(self):
        self.assertFalse(prompt_route([False, False]))

    def test_compatible_prompt_batch_uses_the_prompt_route(self):
        self.assertTrue(prompt_route([True, True]))

    def test_mixed_batch_requires_a_separate_dependency_design(self):
        with self.assertRaises(ValueError):
            prompt_route([True, False])

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            prompt_route([])


if __name__ == '__main__':
    unittest.main()
