"""The eager control cannot silently weaken a normal serving baseline."""
from pathlib import Path
import unittest

from .servers import command


class EagerDiagnosticTest(unittest.TestCase):
    def test_only_explicit_token_correctness_uses_eager(self):
        for role in ['prefill','decode']:
            normal=command(role,18841,Path('/dev/shm/cpu_only_command'),19841,
                observe=True,token_split=True)
            diagnostic=command(role,18841,Path('/dev/shm/cpu_only_command'),19841,
                observe=True,token_split=True,eager_reference=True)
            self.assertNotIn('--enforce-eager',normal)
            self.assertIn('--enforce-eager',diagnostic)
        with self.assertRaises(AssertionError):
            command('agg',18841,Path('/dev/shm/cpu_only_command'),19841,
                observe=True,eager_reference=True)
        with self.assertRaises(AssertionError):
            command('prefill',18841,Path('/dev/shm/cpu_only_command'),19841,
                observe=True,eager_reference=True)


if __name__=='__main__':unittest.main()
