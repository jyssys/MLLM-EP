import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from .common import ROOT, REPO
from .publish import check_prose_whitespace


class PublicationWhitespaceTests(unittest.TestCase):
    def test_only_the_preserved_checkpoint_can_retain_blank_eof(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)

            def git(*args, env=None):
                return subprocess.check_output(['git', *args], cwd=folder, env=env,
                    text=True, stderr=subprocess.STDOUT).strip()

            git('init', '-q')
            git('-c', 'user.name=CPU Test', '-c', 'user.email=test@example.invalid',
                'commit', '--allow-empty', '-qm', 'Fixture')
            frozen = str((ROOT / 'campaign_resume_checkpoint.md').relative_to(REPO))
            checkpoint = folder / frozen
            checkpoint.parent.mkdir(parents=True)
            original = b'Frozen evidence checkpoint.\n\n'
            checkpoint.write_bytes(original)
            ordinary = folder / 'new.py'
            ordinary.write_text('value = 1\n')
            git('add', '--', frozen, 'new.py')
            with self.assertRaises(subprocess.CalledProcessError):
                git('diff', '--cached', '--check', 'HEAD', '--', frozen, 'new.py')
            check_prose_whitespace(git, 'HEAD', [frozen, 'new.py'], dict(os.environ))
            self.assertEqual(checkpoint.read_bytes(), original)
            ordinary.write_text('value = 1\n\n')
            git('add', '--', 'new.py')
            with self.assertRaises(subprocess.CalledProcessError):
                check_prose_whitespace(git, 'HEAD', [frozen, 'new.py'], dict(os.environ))


if __name__ == '__main__':
    unittest.main()
