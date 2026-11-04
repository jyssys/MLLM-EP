import tempfile
import unittest
from pathlib import Path
from .common import write
from .initial_qualification import reusable_qualification


class CacheTests(unittest.TestCase):
    def test_reuse_requires_the_same_reference_and_immutable_config(self):
        with tempfile.TemporaryDirectory(prefix='qualification_cache_test_') as temporary:
            folder = Path(temporary)
            cfg = dict(name='example', count=32)
            write(folder / 'configuration.json', cfg)
            write(folder / 'dispatcher_result.json', dict(complete=True))
            write(folder / 'qualification.json', dict(complete=True, passed=True,
                output=dict(reference='native'), state=dict(reference='native')))
            self.assertTrue(reusable_qualification(folder, cfg, 'native')['passed'])
            write(folder / 'arc_contract.json', cfg)
            write(folder / 'configuration.json', dict(arguments=dict(model='example')))
            self.assertTrue(reusable_qualification(folder, cfg, 'native')['passed'])
            self.assertIsNone(reusable_qualification(folder, cfg, 'other_native'))
            with self.assertRaisesRegex(AssertionError, 'Immutable'):
                reusable_qualification(folder, dict(cfg, count=8), 'native')
            write(folder / 'dispatcher_result.json', dict(complete=False))
            self.assertIsNone(reusable_qualification(folder, cfg, 'native'))


if __name__ == '__main__':
    unittest.main()
