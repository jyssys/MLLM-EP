"""CPU-only definition and safety checks. Not pretrained correctness."""
import json
import tempfile
import unittest
from pathlib import Path
from .common import ROOT, UUIDS, write, digest
from .dispatch import completed

class ContractTests(unittest.TestCase):
    def test_eight_unique_devices(self):
        self.assertEqual(set(UUIDS),set(range(8)))
        self.assertEqual(len(set(UUIDS.values())),8)

    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.json'
            write(path,dict(unit='ns',domain='monotonic',value=23))
            self.assertEqual(json.loads(path.read_text())['value'],23)
            self.assertEqual(len(digest(path)),64)
            self.assertEqual(list(Path(folder).glob('*.tmp')),[])

    def test_registration_pairs(self):
        plan=json.loads((ROOT/'p0_contention_plan.json').read_text())
        jobs=[j for g in plan['groups'] for j in g['jobs']]
        self.assertEqual(len(set(j['id'] for j in jobs)),len(jobs))
        for gpu in range(8):
            self.assertEqual({j['parallel_workers'] for j in jobs if j['gpu']==gpu},{1,2,4,8})
            self.assertEqual({j['startup'] for j in jobs if j['gpu']==gpu},{0,1})

if __name__=='__main__':
    unittest.main()
