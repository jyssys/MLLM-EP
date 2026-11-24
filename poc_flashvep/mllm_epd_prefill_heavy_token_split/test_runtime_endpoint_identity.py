"""Check real local port conflicts and exact HTTP identity before accepting readiness."""
import io
import json
import socket
from types import SimpleNamespace
import unittest

from .runtime_endpoint_identity import require_identity,require_free_ports


class IdentityTest(unittest.TestCase):
    def reply(self,name):
        stream=io.BytesIO(json.dumps(dict(name=name)).encode());stream.status=200
        return stream

    def test_matching_name_still_requires_the_new_process_to_be_alive(self):
        with self.assertRaises(AssertionError):
            require_identity(self.reply('actual'), 'actual',SimpleNamespace(poll=lambda:1))
        value=require_identity(self.reply('actual'),'actual',SimpleNamespace(poll=lambda:None))
        self.assertEqual(value['name'],'actual')

    def test_existing_listener_is_not_treated_as_a_free_port(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1',0));listener.listen(1)
            port=listener.getsockname()[1]
            with self.assertRaises(AssertionError):require_free_ports([port])
        require_free_ports([port])


if __name__=='__main__':unittest.main()
