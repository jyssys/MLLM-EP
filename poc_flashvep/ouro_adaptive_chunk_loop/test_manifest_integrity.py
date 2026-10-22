"""CPU integrity checks. Synthetic files never enter measured result tables."""
import json
from pathlib import Path
import tempfile
import unittest
from .common import ROOT,write,digest
from . import dispatch

class IntegrityTests(unittest.TestCase):
    def test_source_families_do_not_cross_partitions(self):
        manifest=json.loads((ROOT/'workload_manifest.json').read_text())
        seen={}
        for row in manifest['cases']:
            seen.setdefault(row['family'],set()).add(row['split'])
        self.assertTrue(all(len(v)==1 for v in seen.values()))
        self.assertEqual({r['split'] for r in manifest['cases']},{'TRAIN','VALIDATION','CALIBRATION','HELDOUT','FINAL_ONLINE'})

    def test_unique_plan_ids_and_authorized_devices(self):
        for name in ('p1_diagnostic_plan.json','p1_quality_plan.json','p1_controlled_plan.json','p2_surface_plan.json'):
            jobs=json.loads((ROOT/name).read_text())['jobs']
            self.assertEqual(len(jobs),len({j['id'] for j in jobs}))
            self.assertTrue(all(j['gpu'] in range(8) for j in jobs))

    def test_completed_job_rejects_corrupted_result(self):
        original=dispatch.ROOT
        with tempfile.TemporaryDirectory() as folder:
            try:
                dispatch.ROOT=Path(folder)
                target=Path(folder)/'raw'/'synthetic_unit'
                write(target/'result.json',dict(evidence='SYNTHETIC_UNIT_ONLY',value=3))
                write(target/'completed.json',dict(artifacts=[dict(path='result.json',sha256=digest(target/'result.json'))]))
                self.assertTrue(dispatch.completed(dict(id='synthetic_unit')))
                write(target/'result.json',dict(evidence='SYNTHETIC_UNIT_ONLY',value=4))
                with self.assertRaises(AssertionError):
                    dispatch.completed(dict(id='synthetic_unit'))
            finally:
                dispatch.ROOT=original

if __name__=='__main__':
    unittest.main()
