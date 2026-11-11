import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from .ar_diagnostic import complete


class StockARReceiptTests(unittest.TestCase):
    def test_only_measured_stock_rows_need_no_warmup_field(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'status.json').write_text(json.dumps(dict(status='COMPLETE')))
            (folder/'AR_worker_exits.json').write_text(json.dumps(dict(passed=True)))
            row = dict(sample_id=0, input_tokens=8000, output_tokens=2, output_ids=[1,2])
            request = folder/'requests.jsonl'
            request.write_text(json.dumps(row)+'\n')
            cfg = dict(count=1, length=8000, cap=2)
            self.assertTrue(complete(folder,cfg))
            row['warmup'] = True
            request.write_text(json.dumps(row)+'\n')
            self.assertFalse(complete(folder,cfg))
            request.write_text('{')
            self.assertFalse(complete(folder,cfg))


if __name__ == '__main__':
    unittest.main()
