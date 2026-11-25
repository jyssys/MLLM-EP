"""Fresh output writes must not cross a symlink into original evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from .figure_regeneration_closeout import allowed_write,compare_bundle


class FigureRegenerationTest(unittest.TestCase):
    def test_original_symlink_write_is_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);sandbox=root/'output';sandbox.mkdir();original=root/'original';original.write_text('frozen')
            link=sandbox/'original';link.symlink_to(original)
            with self.assertRaises(PermissionError):allowed_write(sandbox,link)
            allowed_write(sandbox,sandbox/'new.csv');self.assertEqual(original.read_text(),'frozen')

    def bundle(self,root,color):
        (root/'figures').mkdir(parents=True)
        Image.new('RGBA',(4,4),color).save(root/'figures/test.png')
        (root/'figures/test.csv').write_text('value\n12.5\n')
        (root/'figures/test.pdf').write_bytes(b'%PDF-1.4\n%%EOF\n')

    def test_pixel_and_CSV_checks_are_separate_from_PDF_render_claim(self):
        with TemporaryDirectory() as folder:
            original=Path(folder)/'original';generated=Path(folder)/'generated'
            self.bundle(original,(1,2,3,255));self.bundle(generated,(1,2,3,255))
            result=compare_bundle(original,generated,'test')
            self.assertTrue(result['passed']);self.assertIn('NOT_MEASURED',result['PDF_pixel_render_verification'])

    def test_alpha_differences_cannot_be_hidden_by_RGB_only_comparison(self):
        with TemporaryDirectory() as folder:
            original=Path(folder)/'original';generated=Path(folder)/'generated'
            self.bundle(original,(1,2,3,255));self.bundle(generated,(1,2,3,254))
            self.assertFalse(compare_bundle(original,generated,'test')['PNG_pixels_exact'])


if __name__=='__main__':unittest.main()
