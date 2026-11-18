"""Catch a modified executed bundle or a mismatched worker bootstrap."""
import hashlib
import io
import unittest
import zipfile
from .validate import frozen_bundle


class BundleValidationTests(unittest.TestCase):
    def bundle(self,adapter,startup):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as archive:
            archive.writestr('poc_flashvep/dense_ar_matched_axis_pd/site_bootstrap.py',adapter)
            archive.writestr('sitecustomize.py',startup)
        data.seek(0)
        return data

    def test_executed_adapter_and_bootstrap_both_match(self):
        source=b'original\n'
        expected={'site_bootstrap.py':hashlib.sha256(source).hexdigest()}
        frozen_bundle(self.bundle(source,source),expected)
        with self.assertRaises(AssertionError):
            frozen_bundle(self.bundle(b'changed\n',source),expected)
        with self.assertRaises(AssertionError):
            frozen_bundle(self.bundle(source,b'changed\n'),expected)


if __name__=='__main__':
    unittest.main()
