import unittest
from collections import deque
from .source_pipeline import OrderedControl


class Channel:
    def __init__(self, messages):
        self.messages = deque(messages)
    def poll(self, *args):
        return bool(self.messages)
    def recv(self):
        return self.messages.popleft()


class OrderedControlTests(unittest.TestCase):
    def test_unrelated_command_stops_the_reader(self):
        channel = OrderedControl(Channel([dict(kind='idle_capacity'),
            dict(kind='abort', request_id='A'), dict(kind='idle_committed', tid='A')]))
        consumed = []
        self.assertEqual(channel.consume_safe({'B'}, consumed.append, 4), 1)
        self.assertEqual(channel.consume_safe({'B'}, consumed.append, 4), 0)
        self.assertEqual(channel.recv()['kind'], 'abort')
        self.assertEqual(channel.recv()['kind'], 'idle_committed')
        self.assertFalse(channel.poll())

    def test_current_request_cannot_change_inside_a_submission(self):
        channel = OrderedControl(Channel([dict(kind='idle_prepared', tid='B')]))
        self.assertEqual(channel.consume_safe({'B'}, self.fail, 4), 0)
        self.assertEqual(channel.recv()['tid'], 'B')

    def test_only_other_generation_scoped_control_is_consumed(self):
        channel = OrderedControl(Channel([dict(kind='idle_prepared', tid='A'),
            dict(kind='idle_committed', tid='A'), dict(kind='idle_rejected')]))
        messages = []
        self.assertEqual(channel.consume_safe({'B'}, messages.append, 8), 2)
        self.assertEqual(channel.recv(), dict(kind='idle_rejected'))

    def test_limit_and_forwarded_control(self):
        channel = OrderedControl(Channel([dict(kind='idle_capacity')]*3))
        messages = []
        self.assertEqual(channel.consume_safe({'B'}, messages.append, 2), 2)
        self.assertTrue(channel.poll())
        self.assertEqual(len(channel.channel.messages), 1)


if __name__ == '__main__':
    unittest.main()
