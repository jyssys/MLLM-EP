"""Sync producer events must count once and retain ownership checks."""
import unittest
from types import SimpleNamespace as S
from .source_events import pending_source_ids


class SourceEventTests(unittest.TestCase):
    def row(self, ready=False, **updates):
        row = dict(outgoing=True, cancelled=False, live=True,
            event=S(query=lambda: ready), request_id='a')
        return dict(row, **updates)

    def test_sync_pending_event_is_visible(self):
        self.assertEqual(pending_source_ids([self.row()]), {'a'})

    def test_native_inflight_is_not_double_counted(self):
        self.assertEqual(pending_source_ids([self.row()], {'a'}), {'a'})

    def test_completed_event_is_not_pending(self):
        self.assertEqual(pending_source_ids([self.row(ready=True)]), set())

    def test_cancelled_stale_and_incoming_owners_are_excluded(self):
        for fields in (dict(cancelled=True), dict(live=False), dict(outgoing=False), dict(event=None)):
            self.assertEqual(pending_source_ids([self.row(**fields)]), set())


if __name__ == '__main__':
    unittest.main()
