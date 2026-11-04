import unittest
from .common import configure
from .dynamic_campaign import serving_configuration, configuration
from .dynamic_recovery import signature


class DynamicServingStartupTests(unittest.TestCase):
    def test_n128_uses_the_same_startup_signature_as_n32_qualification(self):
        configure()
        for model in ('ouro', 'huginn'):
            for axis in ('token', 'loop'):
                with self.subTest(model=model, axis=axis):
                    label = 'token_partial' if axis == 'token' else 'loop_partial'
                    task = dict(model=model, axis=axis, label=label,
                        arm='Token_Dynamic' if axis == 'token' else 'Loop_Dynamic', contract={})
                    qualified = configuration(model, label, {}, 32, 'unused', True)
                    serving = serving_configuration(task, 'heavy')
                    self.assertEqual({key: qualified.get(key) for key in signature()},
                                     {key: serving.get(key) for key in signature()})
                    self.assertFalse(serving.get('prime_dynamic_split_geometries'))
                    self.assertEqual(serving['count'], 128)
                    self.assertTrue(serving['dynamic'])


if __name__ == '__main__':
    unittest.main()
