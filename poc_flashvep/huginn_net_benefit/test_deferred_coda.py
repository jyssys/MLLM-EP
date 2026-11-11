import unittest
from types import SimpleNamespace as S
from .deferred_coda import pending_request_ids, collect_completed


class PendingCodaTests(unittest.TestCase):
    def ticket(self, *ids):
        return S(batch=S(requests=[S(request=S(request_id=key)) for key in ids]))

    def test_empty_queue_does_not_block_requests(self):
        self.assertEqual(pending_request_ids([]), set())

    def test_all_members_of_a_batch_stay_blocked(self):
        self.assertEqual(pending_request_ids([self.ticket('a', 'b')]), {'a', 'b'})

    def test_ticket_identity_is_not_request_identity(self):
        self.assertEqual(pending_request_ids([self.ticket('a'), self.ticket('b')]), {'a', 'b'})

    def test_unready_output_does_not_update_native_state(self):
        ticket = self.ticket('a')
        ticket.ready = lambda: False
        ticket.collect = lambda: self.fail('An unready token cannot be collected')
        outputs, pending, count = collect_completed([ticket], lambda *args: self.fail('No token exists yet'))
        self.assertEqual((outputs, pending, count), ([], [ticket], 0))

    def test_completed_output_uses_native_update_once(self):
        ticket = self.ticket('a')
        ticket.ready = lambda: True
        ticket.collect = lambda: 'real-token-result'
        calls = []
        def native(batch, value):
            calls.append((batch, value))
            return ['delivered']
        outputs, pending, count = collect_completed([ticket], native)
        self.assertEqual((outputs, pending, count), (['delivered'], [], 1))
        self.assertEqual(len(calls), 1)
        self.assertEqual(collect_completed(pending, native), ([], [], 0))
        self.assertEqual(len(calls), 1)

    def test_cancelled_identity_is_left_to_native_update(self):
        ticket = self.ticket('stale-request')
        ticket.ready = lambda: True
        ticket.collect = lambda: 'completed-gpu-result'
        self.assertEqual(collect_completed([ticket], lambda batch, value: []), ([], [], 1))

    def test_collection_failure_propagates_without_clearing_input(self):
        ticket = self.ticket('a')
        ticket.ready = lambda: True
        def failure():
            raise RuntimeError('GPU error')
        ticket.collect = failure
        pending = [ticket]
        with self.assertRaisesRegex(RuntimeError, 'GPU error'):
            collect_completed(pending, lambda *args: [])
        self.assertEqual(pending, [ticket])


if __name__ == '__main__':
    unittest.main()
