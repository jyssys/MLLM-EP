import unittest
from .transport_timings import observation


class TransportObservationTests(unittest.TestCase):
    def test_post_handler_reversal_is_not_clamped_or_called_DMA(self):
        events = [dict(tid='x', event='COORDINATOR_received', role='decode', time=1.01)]
        owners = [dict(tid='x', event='NATIVE_ACK_RECEIVED', role='prefill', time=1.),
                  dict(tid='x', event='NATIVE_CONNECTOR_FORGET', role='prefill', time=1.02)]
        value = observation(events, owners, 'x')
        self.assertAlmostEqual(value['D_received_receipt_to_source_ACK_observation_ms'], -10.)
        self.assertAlmostEqual(value['source_ACK_to_retirement_ms'], 20.)
        self.assertIsNone(value['commit_receipt_to_D_received_receipt_ms'])


if __name__ == '__main__':
    unittest.main()
