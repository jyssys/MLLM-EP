"""Final publication must exclude original tensors and unrelated report names."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import publication_final as module


class FinalPublicationTest(unittest.TestCase):
    def test_raw_pack_retains_failed_structured_raw_but_not_large_tensor_or_symlink(self):
        with TemporaryDirectory() as directory:
            repo=Path(directory);root=repo/'evidence';raw=root/'raw';raw.mkdir(parents=True)
            for name in ['failed.jsonl','closed.csv','prior.json','state.pickle','original.safetensors']:
                (raw/name).write_bytes(b'original')
            (raw/'link.json').symlink_to(raw/'prior.json')
            with patch.object(module,'REPO',repo):
                found=module.unarchived_raw(root,{'evidence/raw/prior.json'})
            self.assertEqual({p.name for p in found},{'failed.jsonl','closed.csv','state.pickle'})

    def test_only_explicit_campaign_report_names_are_inferred(self):
        with TemporaryDirectory() as directory:
            source=Path(directory);(source/'report.py').write_text("name='mllm_epd_observed_v1.md'\nother='dense_ar_frozen.md'\n")
            names=module.report_names(source)
            self.assertIn('mllm_epd_observed_v1.md',names)
            self.assertNotIn('dense_ar_frozen.md',names)


if __name__=='__main__':unittest.main()
