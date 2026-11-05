import unittest
from .handoff_receipts import intervals, continuation_gaps


class HandoffReceiptTests(unittest.TestCase):
    def test_filters_startup_and_keeps_negative_observer_order(self):
        events = [dict(tid='old', request_id='warm', event='COORDINATOR_prefill_started', time=0),
                  dict(tid='new', request_id='request', event='COORDINATOR_prefill_started', time=2),
                  dict(tid='new', role='prefill', event='COORDINATOR_prefill_complete', time=3),
                  dict(tid='new', role='prefill', event='COORDINATOR_commit', time=4),
                  dict(tid='new', role='decode', event='COORDINATOR_received', time=5),
                  dict(tid='new', role='prefill', event='COORDINATOR_released', time=6)]
        owners = [dict(tid='new', role='prefill', event='NATIVE_ACK_RECEIVED', time=4.999),
                  dict(tid='new', role='prefill', event='NATIVE_CONNECTOR_FORGET', time=5.5),
                  dict(tid='new', role='decode', event='NATIVE_CONNECTOR_FORGET', time=9)]
        rows = intervals(events, owners, 1, 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['received_receipt_to_P_ACK_ms'], -1)
        self.assertAlmostEqual(rows[0]['P_complete_receipt_to_P_credit_release_ms'], 3000)
        self.assertAlmostEqual(rows[0]['P_ACK_to_P_forget_receipt_ms'], 501)

    def test_missing_measurement_receipts_do_not_become_zero_latency(self):
        with self.assertRaises(AssertionError):
            intervals([], [], 1, 1)

    def test_continuation_queue_and_delivery_are_not_device_compute(self):
        record = dict(first_token=2., D_continuation_to_first_token_ms=200.,
            D_prompt_device_enclosure_ms=100., actual_P_model_stream_entry=dict(time=1.),
            P_prompt_device_enclosure_ms=500.)
        values = continuation_gaps(record, 1.7)
        self.assertAlmostEqual(values['received_receipt_to_D_prompt_entry_ms'], 100.)
        self.assertAlmostEqual(values['P_model_stream_end_to_D_prompt_entry_ms'], 300.)
        self.assertAlmostEqual(values['D_prompt_end_to_first_token_ms'], 100.)

    def test_native_missing_continuation_remains_unmeasured(self):
        self.assertTrue(all(value is None for value in continuation_gaps(None, 1).values()))


if __name__ == '__main__':
    unittest.main()
