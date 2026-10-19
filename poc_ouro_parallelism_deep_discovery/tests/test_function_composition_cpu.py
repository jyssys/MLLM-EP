import unittest
from fractions import Fraction

from poc_ouro_parallelism_deep_discovery.scripts.function_composition_screen import (
    Affine, balanced_affine, balanced_opaque, compose,
    exhaustive_table_log2_bytes, naive_affine_counts,
)


class Tests(unittest.TestCase):
    def setUp(self):
        self.a = Affine(((Fraction(2), Fraction(1)), (Fraction(0), Fraction(3))),
                        (Fraction(1), Fraction(-1)))
        self.b = Affine(((Fraction(0), Fraction(4)), (Fraction(5), Fraction(1))),
                        (Fraction(-2), Fraction(7)))
        self.x = (Fraction(1, 3), Fraction(2, 7))

    def test_affine_composition_is_exact(self):
        self.assertEqual(compose(self.b, self.a).apply(self.x), self.b.apply(self.a.apply(self.x)))

    def test_composition_order_matters(self):
        self.assertNotEqual(compose(self.b, self.a).apply(self.x), compose(self.a, self.b).apply(self.x))

    def test_affine_associativity(self):
        self.assertEqual(compose(self.a, compose(self.b, self.a)), compose(compose(self.a, self.b), self.a))

    def test_balanced_four_maps(self):
        maps = [self.a, self.b, self.a, self.b]
        folded, levels = balanced_affine(maps)
        result = self.x
        for item in maps:
            result = item.apply(result)
        self.assertEqual(folded.apply(self.x), result)
        self.assertEqual(levels, 2)

    def test_opaque_construction_does_not_evaluate(self):
        events = []
        callback, levels = balanced_opaque([1, 2, 3, 4], events)
        self.assertEqual(events, [])
        self.assertEqual(levels, 2)
        self.assertIsInstance(callback(1), int)
        self.assertEqual([row['leaf'] for row in events], [1, 2, 3, 4])
        self.assertTrue(all(a['output'] == b['input'] for a, b in zip(events, events[1:])))

    def test_counted_fold_is_not_free(self):
        counts = naive_affine_counts(2, 4)
        self.assertEqual((counts['serial_multiplications'], counts['folded_multiplications']), (16, 40))
        self.assertEqual((counts['serial_additions'], counts['folded_additions']), (16, 28))

    def test_full_table_size_is_representation_specific(self):
        self.assertEqual(exhaustive_table_log2_bytes(2048), 32780)
        self.assertEqual(exhaustive_table_log2_bytes(1), 17)

    def test_invalid_inputs(self):
        for func, args in [(balanced_affine, []), (balanced_opaque, [])]:
            with self.assertRaises(ValueError):
                func(args) if func is balanced_affine else func(args, [])
        with self.assertRaises(ValueError):
            self.a.apply((1,))
        with self.assertRaises(ValueError):
            Affine(((1,),), (1, 2))
        with self.assertRaises(ValueError):
            naive_affine_counts(0, 4)
        with self.assertRaises(ValueError):
            exhaustive_table_log2_bytes(3)


if __name__ == '__main__':
    unittest.main()
