"""A blocked model call must not block a host-only credit response."""
import multiprocessing
import threading
import unittest
from .credit_fast_lane import ControlChannel, locked


class FastControlTests(unittest.TestCase):
    def test_response_without_main_worker_poll(self):
        parent, child = multiprocessing.Pipe()
        channel = ControlChannel(child, lambda command: dict(kind='reservation_result', generation=command['generation']))
        parent.send(dict(kind='reserve', generation='g1'))
        self.assertTrue(parent.poll(1.))
        self.assertEqual(parent.recv()['generation'], 'g1')
        parent.send(dict(kind='add', request_id='q0'))
        self.assertEqual(channel.recv()['request_id'], 'q0')
        parent.send(dict(kind='stop'))
        self.assertEqual(channel.recv()['kind'], 'stop')
        channel.close()
        parent.close()

    def test_original_connection_has_no_automatic_credit_response(self):
        parent, child = multiprocessing.Pipe()
        parent.send(dict(kind='reserve', generation='g1'))
        self.assertFalse(parent.poll(.01))
        self.assertEqual(child.recv()['generation'], 'g1')
        parent.close()
        child.close()

    def test_lock_preserves_atomic_counter(self):
        lock = threading.RLock()
        value = [0]
        def increment():
            before = value[0]
            value[0] = before + 1
        operation = locked(lock, increment)
        threads = [threading.Thread(target=lambda: [operation() for _ in range(5000)]) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(value[0], 20000)


if __name__ == '__main__':
    unittest.main()
