"""Absent artifacts remain absent. A file check does not establish science gates."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .delivery_requirement_inventory import figure_bundle,FIGURES


class DeliveryInventoryTest(unittest.TestCase):
    def test_missing_bundle_does_not_fabricate_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            row=figure_bundle(Path(folder),'missing')
            self.assertFalse(row['available']);self.assertEqual(len(row['missing']),3)

    def test_image_PDF_and_external_CSV_are_all_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'figures').mkdir();(root/'analysis').mkdir()
            Image.new('RGB',(4,4)).save(root/'figures/test.png')
            (root/'figures/test.pdf').write_bytes(b'%PDF-1.4\nnot a full parsed PDF\n%%EOF\n')
            (root/'analysis/windows.csv').write_text('duration_ms\n1.25\n')
            row=figure_bundle(root,'test','analysis/windows.csv')
            self.assertTrue(row['available']);self.assertEqual(row['CSV_rows'],1)
            self.assertEqual(row['PNG_dimensions'],[4,4]);self.assertEqual(len(row['files']),3)
            self.assertIn('Not a semantic or full PDF-render audit',row['checks'])

    def test_all_requested_figure_identifiers_are_unique(self):
        keys=[r[0] for r in FIGURES]
        self.assertEqual(len(keys),26);self.assertEqual(len(set(keys)),26)
        self.assertTrue(any('OFFLINE' in r[3] for r in FIGURES))


if __name__=='__main__':unittest.main()
