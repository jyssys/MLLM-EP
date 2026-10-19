import datetime as dt
from pathlib import Path
import tempfile
import unittest
from poc_ouro_parallelism_deep_discovery.scripts.validate_final_dossier import (
    recorded_duration, local_link_targets, candidate_scope)


class Tests(unittest.TestCase):
    def row(self, start=0, end=60, actor='coordinator'):
        base=dt.datetime(2026,10,19,tzinfo=dt.timezone.utc)
        return dict(start_utc=(base+dt.timedelta(seconds=start)).isoformat(),
                    end_utc=(base+dt.timedelta(seconds=end)).isoformat(),
                    elapsed_seconds=end-start,actor=actor,evidence=['synthetic'])

    def duration(self, rows):
        return recorded_duration(rows, dt.datetime(2026,10,20,tzinfo=dt.timezone.utc))

    def test_counts_only_coordinator(self):
        self.assertEqual(self.duration([self.row(),self.row(actor='worker')])['seconds'],60)

    def test_excludes_pause_gap(self):
        self.assertEqual(self.duration([self.row(),self.row(120,180)])['seconds'],120)

    def test_rejects_overlap(self):
        with self.assertRaises(ValueError):self.duration([self.row(),self.row(59,120)])

    def test_rejects_future(self):
        with self.assertRaises(ValueError):self.duration([self.row(100000,100060)])

    def test_rejects_wrong_duration(self):
        row=self.row();row['elapsed_seconds']=61
        with self.assertRaises(ValueError):self.duration([row])

    def test_legacy_rounded_duration_keeps_exact_endpoint_sum(self):
        row=self.row(end=60.123456);row['elapsed_seconds']=60.123
        result=self.duration([row])
        self.assertEqual(result['seconds'],60.123456)
        self.assertEqual(result['reported_seconds'],60.123)

    def test_rounding_is_not_an_arbitrary_tolerance(self):
        row=self.row(end=60.123456);row['elapsed_seconds']=60.124
        with self.assertRaises(ValueError):self.duration([row])

    def test_rejects_nonfinite_reported_duration(self):
        row=self.row();row['elapsed_seconds']=float('nan')
        with self.assertRaises(ValueError):self.duration([row])

    def test_rejects_missing_evidence(self):
        row=self.row();row['evidence']=[]
        with self.assertRaises(ValueError):self.duration([row])

    def test_rejects_naive_time(self):
        row=self.row();row['start_utc']='2026-10-19T00:00:00'
        with self.assertRaises(ValueError):self.duration([row])

    def test_local_links_preserve_space_and_strip_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.md'
            path.write_text('[local](<a file.md:12>) [fragment](#a) [web](https://example.com/a) [other](next.md#section)')
            self.assertEqual(local_link_targets(path),[Path(directory)/'a file.md',Path(directory)/'next.md'])

    def test_candidate_scope_requires_unique19(self):
        rows=[dict(id=f'C{i:02}',status='GENERIC_ONLY',candidate_live_E2E='NOT_RUN',
                   candidate_free_running_quality='NOT_RUN') for i in range(1,20)]
        self.assertEqual(candidate_scope(rows)['count'],19)
        with self.assertRaises(ValueError):candidate_scope(rows[:-1]+[rows[0]])

    def test_candidate_cannot_gain_measured_label(self):
        rows=[dict(id=f'C{i:02}',status='GENERIC_ONLY',candidate_live_E2E='NOT_RUN',
                   candidate_free_running_quality='NOT_RUN') for i in range(1,20)]
        rows[0]['candidate_live_E2E']='PASS'
        with self.assertRaises(ValueError):candidate_scope(rows)


if __name__ == '__main__':unittest.main()
